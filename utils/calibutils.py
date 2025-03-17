"""Main calibration routine for SWMM models."""

import os
import time
import uuid
import shutil
import pickle
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import yaml
import swmmio
from swmmio import Model
from swmm.toolkit.shared_enum import SubcatchAttribute, NodeAttribute, LinkAttribute
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm
from utils.swmmutils import get_node_timeseries



from utils.functions import sync_timeseries, invert_dict, load_dict
from utils.networkutils import get_upstream_nodes, get_downstream_nodes
from utils.swmmutils import get_model_path, run_swmm, dataframe_to_dat, get_predictions_at_nodes
from utils.calibconstraints import CalParam, get_cal_params, get_calibration_order

from defs import CALIBRATION_ROUTINES, CALIBRATION_FORCINGS, ALGORITHMS

from multiprocessing import freeze_support

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt

# Load conceptual SWMM model
model_dir = Path('dat/winnipeg1')
inp_file = Path(os.path.join(model_dir, 'conceptual_model1.inp'))

# Load IW-SWMM node ID mapping
# Comparison nodes comprised of SWMM junctions and outfalls
from utils.optconfig import OptConfig

def calibrate(opt_config: Path | str | OptConfig):
    """
    Complete calibration routine for a SWMM model.

    :param opt_config: Path to the optimization configuration file or an OptConfig object.
    :type opt_config: Path, str, or OptConfig
    :returns: None
    """

    #load the optimization configuration
    if isinstance(opt_config, (str, Path)):
        opt = OptConfig(opt_config)
    elif isinstance(opt_config, OptConfig):
        opt = opt_config
    else:
        raise ValueError("opt_config_file must be a path to a yaml file or an OptConfig object")
    
    opt._standardize_config()

    # Load calibration data
    cal_forcings, cal_targets = load_calibration_data(routine=opt.cfg["routine"])

    model_dir = Path(opt.cfg["base_model_dir"])

    if not opt.cfg["calibration_dir"].is_dir():
        copytree(model_dir, str(opt.cfg["calibration_dir"]))

    #copy the base directory to the calibration directory
    if not (opt.cfg["calibration_dir"] / "runs").is_dir():
        os.mkdir(str(opt.cfg["calibration_dir"] / "runs"))

    #copy the base model in the calibration directory; this file will be modified inplace
    cal_model = opt.cfg["calibration_dir"] / f"{opt.cfg['routine']}_CALIBRATION.inp"
    if cal_model.exists():
        os.remove(cal_model)

    shutil.copyfile(str(model_dir / opt.cfg["base_model_name"]), cal_model)
    
    if not (opt.cfg["calibration_dir"] / opt.cfg["base_model_name"]).exists():
        warnings.warn(f"Base model {opt.cfg['base_model_name']} not found in {opt.cfg['calibration_dir']}, copying from {opt.cfg['base_model_dir']}")
        shutil.copyfile(Path(opt.cfg['base_model_dir']) / opt.cfg['base_model_name'], opt.cfg["calibration_dir"] / opt.cfg['base_model_name'])
    

    cal_model = Model(cal_model)

    #initialize the run directory within the calibration directory
    # each run has its own timestamped directory within the 'run' folder of the calibration directory
    run_dir = initialize_run(Path(os.path.join(opt.cfg["calibration_dir"], 'runs')), opt.cfg["routine"])

    #save the calibration configuration to the run directory
    opt.save_config(output_path=Path(os.path.join(run_dir, 'opt_config.yaml')))
    
    #begin the calibration loop
    logger = initialize_logger()

    detailed_score_file = Path(os.path.join(run_dir, 'detailed_scores.txt'))
    if not detailed_score_file.exists():
        with open(detailed_score_file, 'a+') as f:
            f.write('datetime,iter,obj_param,node,score\n')

    detailed_param_file = Path(os.path.join(run_dir, 'detailed_params.txt'))
    if not detailed_param_file.exists():
        with open(detailed_param_file, 'a+') as f:
            f.write('datetime,iter,name,init_val,cal_val\n')
        
    logger.info("Calibration started for routine: {routine}")

    if opt.cfg["calibration_nodes"] == []:
        comparison_nodes_swmm = cal_model.inp.junctions.index.to_list()  # + model.inp.outfalls.index.to_list()
    else:
        comparison_nodes_swmm = opt.cfg["calibration_nodes"]

    cal_params = get_cal_params(opt.cfg["routine"], cal_model)
    params = {'param':'value'}
    iterations = {'element':'node, score, n_fun_evals, duration'}

    # check that the config target variables are in the loaded calibration targets
    target_keys = set(cal_targets.keys())
    config_keys = set(opt.cfg["target_variables"].keys())
    if not config_keys.issubset(target_keys):
        raise ValueError(f"Calibration targets {cal_targets.keys()} not in {opt.cfg['target_variables'].keys()}")
    
    # select only the targets listed in the config file
    if opt.cfg["target_variables"] != {}:
        cal_targets = {key:cal_targets[key] for key in cal_targets if key in list(opt.cfg["target_variables"].keys())}

    logger.info("Calibration data loaded, forcings are {cal_forcings.keys()} and targets are {cal_targets.keys()}")
    logger.info("Starting calibration loop...")
    
    # overwrite rainfall file with calibration forcings
    filename = Path(os.path.join(get_model_path(cal_model, as_str=False).resolve().parent , 'precip.dat'))
    dataframe_to_dat(filename,cal_forcings["precip"])
    
    # set simulation datetimes
    if opt.cfg["start_date"] == 'None':
        dti = get_shared_datetimeindex(cal_forcings|cal_targets)
        start_time, end_time = dti[0], dti[-1]
        msg = f"start_date and end_date not found in config, using the datetime index of the calibration data, using {start_time} to {end_time}."
    else:
        start_time = datetime.strptime(opt.cfg["start_date"], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(opt.cfg["end_date"], "%Y-%m-%d %H:%M:%S")

    logger.info(f"start_date and end_date found in config, using {start_time} to {end_time}.")

    set_simulation_datetime(
        model=cal_model, 
        start_time=start_time, 
        end_time=end_time
    ).inp.save()

    
    counter = OptCount()
    
    if opt.cfg["hierarchial"]:

     
        #df = get_calibration_order(cal_params, cal_model)
        #df = df[(df["node"] == "node_18") | (df["node"] == "None")]

        cal_params = get_calibration_order(cal_params, cal_model)

        # remove nodes that aren't included among 'comparison nodes'
        # this doesn't generalize super well, as there may be cases you'd still want to calibrate a node that isn't in the comparison nodes
        # however in our case, this only includes the dummy-outfalls, since we have obs everywhere
        cal_params = [c for c in cal_params if c.node in comparison_nodes_swmm]


        # concatenate level 0 elements (those that are not associated with a specific node)
        # this will result in a calibration pass prior, and after, the node-specific calibration
        #levels = np.concatenate([levels,[0]])

        # sort the calibration parameters by level
        levels = [c.level for c in cal_params]
        levels = np.unique(levels)
        levels = np.sort(levels)

        

        #calibration_order = np.argsort(levels)
        #cal_params = [cal_params[i] for i in calibration_order]

        #for cal_param in cal_params:
        for level in levels:
            cal_param_subset = [c for c in cal_params if c.level == level]
            node_subset = np.unique([c.node for c in cal_param_subset]).tolist()

            #node_subset = [node for node in node_subset if node in opt.cfg["calibration_nodes"]]



            if "all" in [c.node.lower() for c in cal_param_subset]:
                node_subset = opt.cfg["calibration_nodes"]

            if len(node_subset) == 0:
                pass
            else:
                cal_model, results = de_calibration(
                    cal_forcings=cal_forcings,
                    cal_targets=cal_targets,
                    in_model=cal_model,
                    eval_nodes=node_subset, 
                    cal_params=cal_param_subset,
                    run_dir=run_dir,
                    opt_config=opt.cfg,
                    counter=counter,
                )

                cal_model.inp.save()
            
            """
            #cal_params_subset = [c for c in cal_params if c.tag in df[df.level == level].element]
            if type(cal_param) is not list:
                cal_param = [cal_param]


            else:
                eval_nodes = [c.node for c in cal_param]
            """

            """
            if np.any(df.loc[df.level == level,"node"] == "None"):
                comparison_nodes_subset = comparison_nodes_swmm
            else:
                comparison_nodes_subset = df.loc[df.level == level,"node"].unique().tolist()

            # remove constraints associated with nodes that are downstream of the comparison nodes
            # since these will have a gradient of 0
            ds_nodes = []
            for node in comparison_nodes_subset:
                ds_nodes = ds_nodes + get_downstream_nodes(cal_model.network, node)[1:]
            
            ds_nodes = np.unique(ds_nodes).tolist()
            cal_params_subset = [c for c in cal_params if c.tag in df[[n not in ds_nodes for n in df.node]].element]

            if len(cal_params_subset) == 0:
                pass
            """
                


    else:
        cal_model, results = de_calibration(
            cal_forcings=cal_forcings,
            cal_targets=cal_targets,
            in_model=cal_model,
            eval_nodes=comparison_nodes_swmm, 
            cal_params=cal_params, 
            run_dir=run_dir,
            opt_config=opt.cfg,
            counter=counter,
        )

        cal_model.inp.save()

    #path_prm = Path(os.path.join("output", 'calib_prm_' + routine + '.txt'))
    #path_res = Path(os.path.join("output", 'calib_res_' + routine + '.txt'))

    #save_dict(filename=path_prm, d=params)
    #save_dict(filename=path_res, d=iterations)

    logger.info("Calibration loop complete.")    


    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


    """reload and run the calibrated model"""
    model = Model(get_model_path(cal_model))
    run_swmm(model)
    model.inp.save()

    """save a copy to the run directory (for archiving purposes)"""
    model.inp.save(str(run_dir / "calibrated_model.inp"))

    """evaluate the calibrated model"""

    """
    sim = get_predictions_at_nodes(model=model, nodes=comparison_nodes_swmm)
    obs = cal_targets[f"flow"].loc[:, comparison_nodes_swmm]
    sim, obs = sync_timeseries(sim, obs)

    # truncate early, unstable portion of simulations
    sim = sim.iloc[48:,:]
    obs = obs.iloc[48:,:]  

    sim.to_csv(Path(os.path.join(run_dir, f"sim_flows_{opt.cfg['routine']}.csv")))
    obs.to_csv(Path(os.path.join(run_dir, f"obs_flows_{opt.cfg['routine']}.csv")))

    # Performance metrics for the calibration
    performance = pf.get_performance(obs, sim)
    metrics = ['mean_obs', 'mean_sim', 'peak_obs', 'peak_sim', 'pve', 'rmse', 'nrmse', 'pep', 'nse', 'mse']
    performance[metrics].to_csv(Path(os.path.join(run_dir, f"performance_{opt.cfg['routine']}.csv")))
    """
    print("\nSaving calibration...", end=" ")
    print("done.")


import logging
import sys
from shutil import copytree


def initialize_logger():
    """
    Initialize a logger object.

    :returns: Logger object
    :rtype: logging.Logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    return logger


def initialize_run(run_loc: Path, name: str):
    """
    Creates a timestamped run directory to track calibration results.

    :param run_loc: Path to the directory where the run directory will be created.
    :type run_loc: Path
    :param name: Name of the calibration routine.
    :type name: str
    :returns: Path to the created run directory.
    :rtype: Path
    """
    now = datetime.now()
    current_time = now.strftime("%d-%m-%y-%H%M%S")
    run_dir = "run-{}_{}".format(name, current_time)
    x = Path(os.path.join(run_loc, run_dir))
    os.mkdir(x)
    return x


def set_params(cal_vals, cal_params, model: Model) -> Model:
    """
    Set parameters in the model based on the values and constraints.

    :param values: Parameter values.
    :type values: list of floats
    :param constraints: Parameter constraints.
    :type constraints: Constraints object
    :param model: Model to be calibrated.
    :type model: swmmio.Model object
    :returns: Model with updated parameters.
    :rtype: swmmio.Model object
    """

    changes = pd.DataFrame(index=[con.tag for con in cal_params], columns=["cal_value","initial_value"])

    for ii in range(len(cal_vals)):

        cp = cal_params[ii]

        bounds = np.array([cp.lower_bound,cp.upper_bound])

        if isinstance(cp.element,tuple):
            index = cp.make_multi_index(model)
        else:
            index = getattr(model.inp, cp.section).index

        idx = index == cp.element

        #swmm_val = getattr(model.inp, con.section).loc[idx,con.attribute]

        #if cp.change == 'relative':
            #new_val = (1+cal_vals[ii]) * cp.initial_value
        #elif cp.change == 'absolute':
        new_val = cal_vals[ii]
       # else:
        #    raise ValueError(f"Change type {cp.change} not recognized")
        
        if cp.attribute in ['CurveNum','PercImperv','Barrels']:
            new_val = int(new_val)


        
        # truncate values within upper and lower bounds
        new_val = truncate_values(new_val, bounds)

        old_val = getattr(model.inp, cp.section).loc[idx,cp.attribute].values[0]
        getattr(model.inp, cp.section).loc[idx,cp.attribute] = new_val

        changes.loc[cp.tag,"cal_value"] = new_val
        changes.loc[cp.tag,"initial_value"] = old_val

    return model, changes


def truncate_values(values, bounds):
    """
    Truncate values to within the specified bounds.

    :param values: Values to be truncated.
    :type values: np.array
    :param bounds: Lower and upper bounds.
    :type bounds: list of floats
    :returns: Truncated values.
    :rtype: np.array
    """
    if type(values) != np.array:
        values = np.array(values)
    values[values < bounds[0]] = bounds[0]
    values[values > bounds[1]] = bounds[1]
    return values


def set_simulation_datetime(model: Model, start_time=None, end_time=None) -> Model:
    """
    Set the simulation start and end times in the model.

    :param model: Model to be updated.
    :type model: swmmio.Model object
    :param start_time: Simulation start time.
    :type start_time: datetime object
    :param end_time: Simulation end time.
    :type end_time: datetime object
    :returns: Model with updated simulation times.
    :rtype: swmmio.Model object
    """
    model.inp.options.loc['START_TIME'] = datetime.strftime(start_time, format='%H:%M:%S')
    model.inp.options.loc['START_DATE'] = datetime.strftime(start_time, format='%m/%d/%Y')
    model.inp.options.loc['REPORT_START_TIME'] = datetime.strftime(start_time, format='%H:%M:%S')
    model.inp.options.loc['REPORT_START_DATE'] = datetime.strftime(start_time, format='%m/%d/%Y')
    model.inp.options.loc['END_TIME'] = datetime.strftime(end_time, format='%H:%M:%S')
    model.inp.options.loc['END_DATE'] = datetime.strftime(end_time, format='%m/%d/%Y')
    return model


def fix_model_strings(model):
    """
    Fix blank timeseries bug in swmmio.

    :param model: Model to be fixed.
    :type model: swmmio.Model object
    :returns: Fixed model.
    :rtype: swmmio.Model object
    """
    for name, row in model.inp.dwf.iterrows():
        timepatterns = model.inp.dwf.loc[name,'TimePatterns'].split(' ')
        model.inp.dwf.loc[name,'TimePatterns'] = ("".join(['"{}"'.format(x) for x in timepatterns]))
    model.inp.inflows['Time Series'] = '""'
    return model


def get_scaler(obs: pd.DataFrame):
    """
    Return mean and std of observed data.

    :param obs: Observed data.
    :type obs: pd.DataFrame
    :returns: Mean and standard deviation of observed data.
    :rtype: tuple
    """
    return (obs.mean(), obs.std())


def normalise(x, scaler):
    """
    Normalise data.

    :param x: Data to be normalised.
    :type x: pd.DataFrame
    :param scaler: Mean and std of data.
    :type scaler: tuple
    :returns: Normalised data.
    :rtype: pd.DataFrame
    """
    return (x - scaler[0]) / scaler[1]


def denormalise(x, scaler):
    """
    Denormalise data.

    :param x: Data to be denormalised.
    :type x: pd.DataFrame
    :param scaler: Mean and std of data.
    :type scaler: tuple
    :returns: Denormalised data.
    :rtype: pd.DataFrame
    """
    return x * scaler[1] + scaler[0]



def truncate_values(values, bounds):
    """
    Truncate values to within the specified bounds.

    :param values: Values to be truncated.
    :type values: np.array
    :param bounds: Lower and upper bounds.
    :type bounds: list of floats
    :returns: Truncated values.
    :rtype: np.array
    """
    if type(values) != np.array:
        values = np.array(values)
    values[values < bounds[0]] = bounds[0]
    values[values > bounds[1]] = bounds[1]
    return values


def set_simulation_datetime(model: Model, start_time=None, end_time=None) -> Model:
    """
    Set the simulation start and end times in the model.

    :param model: Model to be updated.
    :type model: swmmio.Model object
    :param start_time: Simulation start time.
    :type start_time: datetime object
    :param end_time: Simulation end time.
    :type end_time: datetime object
    :returns: Model with updated simulation times.
    :rtype: swmmio.Model object
    """
    model.inp.options.loc['START_TIME'] = datetime.strftime(start_time, format='%H:%M:%S')
    model.inp.options.loc['START_DATE'] = datetime.strftime(start_time, format='%m/%d/%Y')
    model.inp.options.loc['REPORT_START_TIME'] = datetime.strftime(start_time, format='%H:%M:%S')
    model.inp.options.loc['REPORT_START_DATE'] = datetime.strftime(start_time, format='%m/%d/%Y')
    model.inp.options.loc['END_TIME'] = datetime.strftime(end_time, format='%H:%M:%S')
    model.inp.options.loc['END_DATE'] = datetime.strftime(end_time, format='%m/%d/%Y')
    return model


def fix_model_strings(model):
    """
    Fix blank timeseries bug in swmmio.

    :param model: Model to be fixed.
    :type model: swmmio.Model object
    :returns: Fixed model.
    :rtype: swmmio.Model object
    """
    for name, row in model.inp.dwf.iterrows():
        timepatterns = model.inp.dwf.loc[name,'TimePatterns'].split(' ')
        model.inp.dwf.loc[name,'TimePatterns'] = ("".join(['"{}"'.format(x) for x in timepatterns]))
    model.inp.inflows['Time Series'] = '""'
    return model


def get_scaler(obs: pd.DataFrame):
    """
    Return mean and std of observed data.

    :param obs: Observed data.
    :type obs: pd.DataFrame
    :returns: Mean and standard deviation of observed data.
    :rtype: tuple
    """
    return (obs.mean(), obs.std())


def normalise(x, scaler):
    """
    Normalise data.

    :param x: Data to be normalised.
    :type x: pd.DataFrame
    :param scaler: Mean and std of data.
    :type scaler: tuple
    :returns: Normalised data.
    :rtype: pd.DataFrame
    """
    return (x - scaler[0]) / scaler[1]


def denormalise(x, scaler):
    """
    Denormalise data.

    :param x: Data to be denormalised.
    :type x: pd.DataFrame
    :param scaler: Mean and std of data.
    :type scaler: tuple
    :returns: Denormalised data.
    :rtype: pd.DataFrame
    """
    return x * scaler[1] + scaler[0]


def copy_temp_file(filename:Path, tag="TEMP_FILE"):
    if type(filename) is str:
        filename = Path(filename)
        
    x = uuid.uuid4()
    tmp = str(filename).split(".")
    tmp2 = tmp[0] + f"-{tag}-" + str(x) + "." + tmp[1]
    cal_model_tmp = Path(tmp2)
    return cal_model_tmp

def de_score_fun(
        values,
        in_model,
        cal_model,
        cal_params,
        cal_targets,
        run_dir,
        eval_nodes:list[str]=None,
        counter=0,
        opt_config=None):
    """
    Optimization score function.

    :param values: Parameter values.
    :type values: list of floats
    :param in_model: Initial model.
    :type in_model: swmmio.Model object
    :param cal_model: Model to be calibrated.
    :type cal_model: swmmio.Model object
    :param constraints: Parameter constraints.
    :type constraints: Constraints object
    :param cal_targets: Calibration targets.
    :type cal_targets: dict
    :param run_dir: Path to the run directory.
    :type run_dir: Path object
    :param warmup_len: Number of warmup timesteps, defaults to 48.
    :type warmup_len: int, optional
    :param eval_nodes: Nodes to evaluate.
    :type eval_nodes: list of str
    :param counter: Optimization counter.
    :type counter: int
    :param log_every_n: Log results every n iterations.
    :type log_every_n: int
    :param target_weights: Weights for calibration targets.
    :type target_weights: dict
    :returns: Optimization score.
    :rtype: float
    """
    # print("DEBUGa")
    
    if run_dir is None:
        raise ValueError("run_dir not found")

    # Retrieve the *base* model
    model = Model(in_model)

    # update model with current parameters 
    # during the optimization algorithm:
    model, changes = set_params(cal_vals=values, cal_params=cal_params, model=model)

    # fix blank timeseries bug in swmmio (converts "" to NaN when reading, 
    # but doesn't write "" to new file), when editing inflows section
    model = fix_model_strings(model)

    # Save the model to a file, then read it again
    # Must save to a file because `run_swmm()` 
    # (and the actual SWMM program) takes a *path*
    # as input argument. 
    
    # Make the file name unique for parallel computing.
    # Add 'TEMPFILE' tag for cleanup later, if needed.
    
    cal_model_tmp = copy_temp_file(cal_model)


    # run SWWM
    model.inp.save(cal_model_tmp)
    run_swmm(Model(cal_model_tmp))
    
    score_df, _ = eval_model(cal_model_tmp, cal_targets, eval_nodes, opt_config=opt_config)
    # weighted sum of multi-objective scores, mean across eval nodes
    # default uniform weights

    # re-invert the score of the NSE outside the minimize function
    #if opt_config["score_function"].lower() == "nse":
    #    score_df = score_df.apply(lambda x: -x)

    if isinstance(cal_targets, list):
        target_weights = np.array([1 for _ in opt_config["target_variables"]])
    elif isinstance(cal_targets, dict):
        target_weights = np.array([opt_config["target_variables"][key] for key in opt_config["target_variables"]])

    if len(cal_targets) != len(target_weights):
        raise ValueError("target_weights must have the same length as cal_targets")

    # convex combination of scores
    target_weights = target_weights / np.sum(target_weights)
    target_weights = target_weights.reshape(1,-1)
    score = np.matmul(target_weights,score_df.loc[list(opt_config["target_variables"].keys()),:].to_numpy()).mean()

    iter = counter.get_count()
    
    detailed_param_file = Path(os.path.join(run_dir, 'detailed_params.txt'))
    detailed_score_file = Path(os.path.join(run_dir, 'detailed_scores.txt'))

    if np.mod(iter,opt_config["log_every_n"]) == 0:
        for node in score_df.columns:
            for tgt in score_df.index:
                now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
                line = f"{now},{iter},{tgt},{node},{score_df.loc[tgt, node]}\n"
                with open(detailed_score_file, 'a+') as f:
                    f.write(line)

    if np.mod(iter,opt_config["log_every_n"]) == 0:
        for name, row in changes.iterrows():
            now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            line = f"{now},{iter},{name},{row['initial_value']},{row['cal_value']}\n"
            with open(detailed_param_file, 'a+') as f:
                f.write(line)

    counter.increment()

    # Clean up, we will not need this model any more
    fname_root = str(cal_model_tmp).split(".")[0]
    os.remove(cal_model_tmp)
    os.remove(Path(fname_root + ".out"))
    os.remove(Path(fname_root + ".rpt"))
    return score






def eval_model(cal_model_tmp, cal_targets, eval_nodes, opt_config):
    """
    Evaluate the model using the calibration targets.

    :param cal_model_tmp: Path to the temporary model file.
    :type cal_model_tmp: Path
    :param cal_targets: Calibration targets.
    :type cal_targets: dict
    :param eval_nodes: Nodes to evaluate.
    :type eval_nodes: list of str
    :param warmup_len: Number of warmup timesteps, defaults to 48.
    :type warmup_len: int, optional
    :param normalize: Whether to normalize the data, defaults to True.
    :type normalize: bool, optional
    :returns: Scores for each calibration target and timeseries results for each calibration target.
    :rtype: tuple[pd.DataFrame, dict]
    """

    if opt_config["score_function"].lower() == "nse":
        score_fun = pf.nse
    elif opt_config["score_function"].lower() == "mse":
        score_fun = pf.mse
    else: 
        raise ValueError(f"Objective function {opt_config['score_function']} not recognized")


    score_df = pd.DataFrame(index=list(cal_targets.keys()), columns=eval_nodes)
    timeseries_results = {"flow": {}, "depth": {}, "tss": {}}

    # TODO: generalize these cases
    if "flow" in list(cal_targets.keys()):
        obs = cal_targets["flow"].loc[:, eval_nodes]

        #sim = get_predictions_at_nodes(model=Model(cal_model_tmp), nodes=eval_nodes, param="FLOW_RATE")
        sim = get_node_timeseries(model=Model(cal_model_tmp),nodes=eval_nodes, params=["TOTAL_INFLOW"])["TOTAL_INFLOW"][eval_nodes]
        sim = sim.resample('15min').mean()

        if opt_config["normalize"]:
            scaler = get_scaler(obs)
            obs = normalise(obs, scaler)
            sim = normalise(sim, scaler)

        obs, sim = sync_timeseries(obs, sim.loc[:, eval_nodes])
        obs = obs.iloc[opt_config["warmup_length"]:, :]
        sim = sim.iloc[opt_config["warmup_length"]:, :]
        score_df.loc["flow", :] = [score_fun(obs.loc[:, col], sim.loc[:, col]) for col in obs.columns]

        if opt_config["normalize"]:
            obs = denormalise(obs, scaler)
            sim = denormalise(sim, scaler)
        timeseries_results["flow"].update({col: {"obs": obs.loc[:, col], "sim": sim.loc[:, col]} for col in obs.columns})

    if "depth" in list(cal_targets.keys()):
        obs = cal_targets["depth"].loc[:, eval_nodes]
        sim = get_node_timeseries(model=Model(cal_model_tmp),nodes=eval_nodes, params=["INVERT_DEPTH"])["INVERT_DEPTH"][eval_nodes]
        sim = sim.resample('15min').mean()

        if opt_config["normalize"]:
            scaler = get_scaler(obs)
            obs = normalise(obs, scaler)
            sim = normalise(sim, scaler)

        obs, sim = sync_timeseries(obs, sim.loc[:, eval_nodes])
        obs = obs.iloc[opt_config["warmup_length"]:, :]
        sim = sim.iloc[opt_config["warmup_length"]:, :]
        score_df.loc["depth", :] = [score_fun(obs.loc[:, col], sim.loc[:, col]) for col in obs.columns]

        if opt_config["normalize"]:
            obs = denormalise(obs, scaler)
            sim = denormalise(sim, scaler)
        timeseries_results["depth"].update({col: {"obs": obs.loc[:, col], "sim": sim.loc[:, col]} for col in obs.columns})


        #timeseries_results["hrt"].update({col: {"obs": obs.loc[:, col], "sim": sim.loc[:, col]} for col in obs.columns})


    # invert sign of NSE since opt function will always minimize
    if opt_config["score_function"].lower() in ["nse"]:
        score_df = score_df.apply(lambda x: -x)

    return score_df, timeseries_results


def de_calibration(in_model,
                   cal_forcings,
                   cal_targets,
                   eval_nodes,
                   cal_params, 
                   run_dir, 
                   opt_config={},
                   counter=None):
    """
    Subroutine for SWMM optimization
    
    PARAMETERS
        in_model: swmmio.Model object
            initial model
        cal_forcings: dict
            calibration forcings
        cal_targets: dict
            calibration targets
        constraints: Constraints object
            parameter constraints
        run_dir: Path object
            path to the run directory
        opt_config: dict
            optimization configuration
    
    RETURNS
        model: swmmio.Model object
            calibrated model
        results: dict
            optimization results
    """
    calibration_start_time = time.time()

    if type(in_model) is Model:
        in_model = get_model_path(in_model,'inp')

    
    opt_fun = lambda x: de_score_fun(x, 
                                     in_model=in_model, 
                                     cal_params=cal_params,
                                     cal_targets = cal_targets,
                                     eval_nodes=eval_nodes, 
                                     cal_model=in_model, 
                                     run_dir=run_dir,
                                     counter=counter,
                                     opt_config=opt_config)
    
    bounds = [(cp.lower, cp.upper) for cp in cal_params]

    if opt_config["algorithm"] == "differential-evolution":
        opt_args = {key.split("diffevol_")[1]: opt_config[key] for key in opt_config if key.startswith("diffevol_")}
        opt_results = differential_evolution(func=opt_fun, bounds=bounds)
    elif opt_config["algorithm"] in ["Nelder-Mead","Powell","CG","BFGS","L-BFGS-B","TNC","COBYLA","SLSQP","trust-constr","dogleg","trust-ncg","trust-exact","trust-krylov"]:
        opt_args = {key.split("minimize_")[1]: opt_config[key] for key in opt_config if key.startswith("minimize_")}
        opt_results = minimize(method=opt_config["algorithm"],fun=opt_fun, bounds=bounds, x0=[c.initial_value + 0.9*(c.initial_value - c.lower_bound) for c in cal_params], options=opt_args)
    else:
        raise NotImplementedError(f"Algorithm {opt_config['algorithm']} not implemented")
    
    model = Model(in_model)
    model, changes = set_params(cal_vals=opt_results.x, cal_params=cal_params, model=model)

    model = fix_model_strings(model)
    model.inp.save()

    score = opt_results.fun
    param_results = {c.tag:x for c, x in zip(cal_params, opt_results.x)}

    if opt_config["algorithm"] in ["differential-evolution"]:
        results = dict()
        results['score'] = -opt_results.fun
        results['params'] = param_results
        results['n_function_evals'] = opt_results['nfev']
        results['total_time'] = time.time() - calibration_start_time
        results['population'] = opt_results['population']
        results['population_energies'] = opt_results['population_energies']
        results['nit'] = opt_results['nit']


    elif opt_config["algorithm"] in ["Nelder-Mead","Powell","CG","BFGS","L-BFGS-B","TNC","COBYLA","SLSQP","trust-constr","dogleg","trust-ncg","trust-exact","trust-krylov"]:
        results = dict()

    # clean up the temporary calibration SWMM files (NOT CURRENTLY USING TEMP DIR BECAUSE SWMMIO NOT HANDLING ABSOLUTE PATHS, MEANS NEED TO COPY PRECIP, HOTSTART, ETC. ON EACH ITERATION)
    """
    for file in [Path(in_model), Path(in_model).with_suffix('.out'), Path(in_model).with_suffix('.rpt')]:
        if file.exists():
            os.remove(file)
    """
    
    return model, results

class OptCount():
    """
    Class container for processing stuff
    """
    _count = 0
    def increment(self):
        """Increment the counter"""
        # Some code here ...
        self._count += 1

    def get_count(self):
        """Return the counter"""
        return self._count

"""
def get_calibration_order(model, constraints) -> list[str]:

    Sorts model elements from upstream to downstream for calibration, based on the networkx graph.

    :param model: SWMM model object.
    :type model: swmmio.Model
    :param include_subcatchments: Include subcatchments in the calibration order.
    :type include_subcatchments: bool
    :param include_conduits: Include conduits in the calibration order.
    :type include_conduits: bool
    :param include_nodes: Include nodes in the calibration order.
    :type include_nodes: bool
    :returns: Elements in the calibration order and nodes in the calibration order.
    :rtype: list of str

    
    nodes = list(model.network.nodes())
    n_upstream_nodes = [len(get_upstream_nodes(model.network, n)) for n in nodes]
    nodes_sorted = np.array(nodes)[np.argsort(n_upstream_nodes)[range(len(nodes))]]
    node_ranks = {n:ii for ii, n in enumerate(nodes_sorted)}

    conduit_ranks = {}
    conduit_to_node2 = {}
    if include_conduits:
        conduit_to_node1 = {key:model.inp.conduits['InletNode'].to_dict()[key] for key in model.inp.conduits['InletNode'].to_dict()}
        conduit_to_node2 = {key:model.inp.conduits['OutletNode'].to_dict()[key] for key in model.inp.conduits['OutletNode'].to_dict()}
        conduit_ranks = {c:(node_ranks[conduit_to_node1[c]]+node_ranks[conduit_to_node2[c]])/2 for c in model.inp.conduits.index}

    subcatchment_ranks = {}
    subcatchment_to_node = {}
    if include_subcatchments:
        subcatchment_to_node = {key:model.inp.subcatchments['Outlet'].to_dict()[key] for key in model.inp.subcatchments['Outlet'].to_dict()}
        subcatchment_ranks = {sc:node_ranks[subcatchment_to_node[sc]]+0.1 for sc in model.inp.subcatchments.index}

    if not include_nodes:
        node_ranks = {}

    elements = list(node_ranks.keys()) + list(conduit_ranks.keys()) + list(subcatchment_ranks.keys())
    ranks = list(node_ranks.values()) + list(conduit_ranks.values()) + list(subcatchment_ranks.values())
    calibration_order = np.array(elements)[np.argsort(ranks)]
 
    if not include_nodes:
        nodes = []
        
    elements = list(subcatchment_to_node.keys()) + list(conduit_to_node2.keys()) + list(nodes)
    nodes = list(subcatchment_to_node.values()) + list(conduit_to_node2.values()) + list(nodes)
    nearest_node = {e:n for e,n in zip(elements, nodes)}
    node_order = [nearest_node[e] for e in calibration_order]
    return calibration_order, node_order
"""

def preprocess_calibration_data(inp_model=Path('dat/winnipeg1/conceptual_model1.inp'), infoworks_dir="data/InfoWorks/exports/sims/batch", output_dir="dat/calibration-data"):
    """
    Preprocess the calibration data for the dry and wet weather calibration routines.

    This is done because loading the infoworks, and calculating HRT is computationally expensive and doesn't need to be done for each calibration run.

    :param inp_model: Path to the SWMM model.
    :type inp_model: Path
    :param infoworks_dir: Directory containing the InfoWorks simulation results.
    :type infoworks_dir: str
    :param output_dir: Directory to save the preprocessed calibration data.
    :type output_dir: str
    :returns: None
    """
    
    model = Model(str(inp_model))
    model_dir = get_model_path(model, as_str=False).parent

    # load IW-SWMM node ID mapping
    # comparison nodes comprised of SWMM junctions and outfalls
    comparison_nodes_swmm = model.inp.junctions.index.to_list()# + model.inp.outfalls.index.to_list()

    filename = Path(os.path.join(model_dir / 'node_id_conversion.json'))
    if not filename.exists():
        raise ValueError("node_id_conversion.json not found in model directory, is should be generated alongside the SWMM model in 'graph-to-swmm.py'")
    node_id_conversion = invert_dict(load_dict(filename))

    fname = open('iwm.pickle', 'rb')
    iwm = pickle.load(fname)
    filename = os.path.join(infoworks_dir, "dwf")
    dwf = iwm.get_predictions_at_nodes(
        Path(filename), 
        nodes=[node_id_conversion[node] for node in comparison_nodes_swmm],
        param="flow")
    dwf.columns = comparison_nodes_swmm
    dwf.to_csv(os.path.join(output_dir,"dry-weather_flow.csv"))
    
    dwf = iwm.get_predictions_at_nodes(
        Path(filename), 
        nodes=[node_id_conversion[node] for node in comparison_nodes_swmm],
        param="depth")
    dwf.columns = comparison_nodes_swmm
    dwf.to_csv(os.path.join(output_dir,"dry-weather_depth.csv"))

    filename = os.path.join(infoworks_dir,"april22")
    
    for param in ["flow","depth","vol"]:
        df = iwm.get_predictions_at_nodes(
            data_dir = Path(filename),
            nodes=[node_id_conversion[node] for node in comparison_nodes_swmm],
            param=param)
        
        df.columns = comparison_nodes_swmm
        df.to_csv(os.path.join(output_dir,f"wet-weather_{param}.csv"))
        
    filename = os.path.join("data","InfoWorks",'exports', 'sims', 'batch',  "april22", "wwf_april22_depth.csv")
    depth_results = get_link_results(filename)

    filename = os.path.join("data","InfoWorks", 'exports', 'sims', 'batch',  "april22", "wwf_april22_flow.csv")
    flow_results = get_link_results(filename)

    A, P = flow_depth_to_AP(
        depth=depth_results.mean().loc[iwm.geom.conduits.index],
        height=iwm.geom.conduits["conduit_height"]/1000,
        width=iwm.geom.conduits["conduit_width"]/1000,
        shapes=iwm.geom.conduits["shape"]
    )

    conduit_hrts = iwm.geom.conduits.loc[:,"conduit_length"] / (flow_results.mean().loc[iwm.geom.conduits.index] / A)

    paths = {}
    for jx in model.inp.junctions.index.tolist():
        # get the corresponding upstream and downstream nodes for each path in the network
        upstream_node = node_id_conversion[jx]
        downstream_node = node_id_conversion[get_downstream_nodes(model.network, jx)[-2]]
        path = nx.dijkstra_path(iwm.graph, upstream_node, downstream_node)
        # if path contains more than one node, get the path conduit IDs
        if len(path) > 1:
            paths[upstream_node] = [iwm.graph.edges[(path[ii], path[ii+1])]["id"] for ii, _ in enumerate(path[:-1])]
        else:
            paths[upstream_node] = []

    hrts = np.array([np.nansum([conduit_hrts.loc[c] for c in paths[path] if c in conduit_hrts.index])/3600 for path in paths])
    hrts[(hrts==0) | (hrts>100)] = np.nan
    hrts = pd.DataFrame(data=hrts, index=model.inp.junctions.index.tolist())
    hrts.to_csv(os.path.join(output_dir,"wet-weather_hrt.csv"))


    folder = os.path.join("data/InfoWorks", 'rainfall')
    file_precip = Path(os.path.join(output_dir, 'precip.pkl') )
    precip_continuous = import_rainfall_folder(folder)
    precip_continuous = pd.DataFrame(precip_continuous)
    precip_continuous.columns = ['rg1']
    precip_continuous.to_pickle(file_precip)



class calibration_data():
    """
    need to know info
    node id
    param id

    target/forcing

    if forcing, make dat
    
    
    """


def load_calibration_data(routine: str, data_dir="dat/calibration-data") -> dict[str: pd.DataFrame]:
    """
    Load the pre-processed calibration data (forcing and observed).

    :param routine: One of ["dry", "wet", "tss"].
    :type routine: str
    :param data_dir: Directory containing the calibration data.
    :type data_dir: str
    :returns: Dictionary containing the calibration forcings and dictionary containing the calibration targets.
    :rtype: dict[str: pd.DataFrame]
    """
    cal_forcings = {}
    cal_targets = {}
    if routine in ["dry"]:
        cal_targets["flow"] = pd.read_csv(os.path.join(data_dir, "dry-weather_flow.csv"), index_col=0, parse_dates=True)
        cal_targets["depth"] = pd.read_csv(os.path.join(data_dir, "dry-weather_depth.csv"), index_col=0, parse_dates=True)

        # create a dummy rainfall dataframe for dry-weather calibration
        one_week = np.arange(datetime(year=2001, month=1, day=1, hour=0, minute=0, second=0), 
                            datetime(year=2001, month=1, day=8, hour=0, minute=0, second=0), 
                            timedelta(minutes=15)).astype(datetime)
        
        dwf_rainfall = pd.DataFrame(index=one_week, columns=['rg1'])
        dwf_rainfall['rg1'] = 0
        cal_forcings["precip"] = dwf_rainfall

    elif routine in ["wet","tss"]:
        cal_targets["flow"] = pd.read_csv(os.path.join(data_dir, "wet-weather_flow.csv"), index_col=0, parse_dates=True)
        cal_targets["depth"] = pd.read_csv(os.path.join(data_dir, "wet-weather_depth.csv"), index_col=0, parse_dates=True)
        cal_targets["vol"] = pd.read_csv(os.path.join(data_dir, "wet-weather_vol.csv"), index_col=0, parse_dates=True)
        cal_targets["hrt"] = pd.read_csv(os.path.join(data_dir, "wet-weather_hrt.csv"), index_col=0)
        cal_targets["hrt"] = cal_targets["hrt"][cal_targets["hrt"].columns[0]] # convert from pd.df to pd.series
        
        cal_forcings["precip"] = pd.read_pickle(os.path.join(data_dir, "precip.pkl"))

        cal_targets["tss"] = preprocess_tss_data(data_dir=Path("data"))
        
    else:
        raise NotImplementedError(f"Routine {routine} is not implemented, choices include: {CALIBRATION_ROUTINES}")
    return cal_forcings, cal_targets


def get_shared_datetimeindex(x: dict[str: pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Get the common datetimeindex of a dictionary of dataframes. Ignores dataframes with non-datetimeindex.

    :param x: A dictionary of dataframes.
    :type x: dict[str: pd.DataFrame]
    :returns: The common datetimeindex of a dictionary of dataframes.
    :rtype: pd.DatetimeIndex
    """
    dti = pd.concat([x[key] for key in x if type(x[key].index) == pd.DatetimeIndex], axis=1).dropna(axis=1, how="all").dropna(axis=0, how="any").index

    return dti




def summarize_runs(runs_dir:Path) -> list[Path]:
    run_dirs = [dir for dir in runs_dir.iterdir() if dir.is_dir()]
    run_dates = [datetime.strptime(d.name.split("_")[1], "%d-%m-%y-%H%M%S") for d in run_dirs]
    df = pd.DataFrame(index=run_dirs, data=run_dates, columns=["dates"])


    for d in run_dirs:
        cfg = load_dict(d / "opt_config.yaml")
        for k, v in cfg.items():
            if v == []:
                v = ''
            if isinstance(v, list):
                v = ', '.join(v)
            if isinstance(v, dict):
                v = ', '.join([f"{k1}: {v1}" for k1, v1 in v.items()])
            df.loc[d, k] = v

        if (not (d/"detailed_scores.txt").exists()) | (not (d/"detailed_params.txt").exists()):
            df.loc[d, "status"] = "failed"
        elif(len(pd.read_csv(d / "detailed_scores.txt")) == 0) | (len(pd.read_csv(d / "detailed_params.txt")) == 0):
            df.loc[d, "status"] = "failed"
        elif not (d/"calibrated_model.inp").exists():
            df.loc[d, "status"] = "incomplete"
        else:
            df.loc[d, "status"] = "complete"

        df = df.sort_values(by="dates")
    return df


