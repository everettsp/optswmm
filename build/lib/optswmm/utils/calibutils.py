"""Main calibration routine for SWMM models."""

# Standard library imports
import logging
import os
import sys
import time
import uuid
import shutil
import pickle
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import contextlib
from multiprocessing import freeze_support

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx
import yaml
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm

# SWMM-related imports
import swmmio
from swmmio import Model
import pyswmm.simulation
from pyswmm import Simulation, Output, Nodes, Links, Subcatchments, SimulationPreConfig
from swmm.toolkit import shared_enum
from swmm.toolkit.shared_enum import SubcatchAttribute, NodeAttribute, LinkAttribute

# Plotting
import matplotlib.pyplot as plt

# Local imports
from optswmm.utils.swmmutils import get_node_timeseries, get_model_path, run_swmm, dataframe_to_dat, get_predictions_at_nodes
from optswmm.utils import perfutils as pf
from optswmm.utils.functions import sync_timeseries, invert_dict, load_dict
from optswmm.utils.networkutils import get_upstream_nodes, get_downstream_nodes
from optswmm.utils.calparams import CalParam, get_cal_params, get_calibration_order
from optswmm.utils.standardization import load_timeseries
from optswmm.utils.optconfig import OptConfig
from optswmm.defs import ROW_ATTRIBUTES, ROW_INDICES, PARAM_INDICES, SWMM_SECTION_SUBTYPES

# Configure warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
MAXIMIZE_FUNCTIONS = ["nse", "kge", "rsr", "rsq", "pearsonr", "spearmanr"]

# Load conceptual SWMM model (consider moving this to a configuration or function)
model_dir = Path('dat/winnipeg1')
inp_file = Path(os.path.join(model_dir, 'conceptual_model1.inp'))

# Load IW-SWMM node ID mapping
# Comparison nodes comprised of SWMM junctions and outfalls


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
    

def calibrate(opt_config: Path | str | OptConfig, cal_params: list[CalParam]):
    """
    Complete calibration routine for a SWMM model.

    :param opt_config: Path to the optimization configuration file or an OptConfig object.
    :type opt_config: Path, str, or OptConfig
    :returns: None
    """

    #load the optimization configuration
    if isinstance(opt_config, Path):
        opt = OptConfig(config_file=opt_config)
    elif isinstance(opt_config, OptConfig):
        opt = opt_config
    else:
        
        raise ValueError("opt_config_file must be a path to a yaml file or an OptConfig object")
    
    opt._standardize_config()
    opt._initialize_run()
    cal_params.to_df().to_csv(opt.run_dir / "calibration_parameters.csv", index=True)
    opt.save_config()
    # Load calibration data
    
    #cal_forcings, cal_targets = load_calibration_data(routine=opt.cfg["routine"])
    

    cal_forcings = load_timeseries(opt.forcing_data_file)
    cal_targets = load_timeseries(opt.target_data_file)


    model_dir = Path(opt.model_file).parent
    model_name = Path(opt.model_file).stem

    #copy the base model in the calibration directory; this file will be modified inplace
    #cal_model = model_dir / f"{model_name}_CALIBRATION.inp"
    #
    #if cal_model.exists():
    #    os.remove(cal_model)

    #shutil.copyfile(opt.model_file, cal_model)

    cal_model = Model(str(opt.model_file))

    #shutil.copyfile(str(model_dir / opt.cfg["base_model_name"]), cal_model)
    
    #initialize the run directory within the calibration directory
    # each run has its own timestamped directory within the 'run' folder of the calibration directory

    #save the calibration configuration to the run directory
    #opt.save_config(output_path=run_dir / 'opt_config.yaml')
    
    #begin the calibration loop
    logger = initialize_logger()


    logger.info(f"Calibration started for model: {opt.model_file.stem}")


    #if opt.cfg["calibration_nodes"] == []:
    #    comparison_nodes_swmm = cal_model.inp.junctions.index.to_list()  # + model.inp.outfalls.index.to_list()
    #else:
    #    comparison_nodes_swmm = opt.cfg["calibration_nodes"]


    params = {'param':'value'}
    iterations = {'element':'node, score, n_fun_evals, duration'}


    # overwrite rainfall file with calibration forcings
    #filename = Path(os.path.join(get_model_path(cal_model, as_str=False).resolve().parent , 'precip.dat'))
    #dataframe_to_dat(filename,cal_forcings["precip"])
    
    # set simulation datetimes
    #if opt.cfg["start_date"] == 'None':
    #    dti = get_shared_datetimeindex(cal_forcings|cal_targets)
    #    start_time, end_time = dti[0], dti[-1]
    #    msg = f"start_date and end_date not found in config, using the datetime index of the calibration data, using {start_time} to {end_time}."
    #else:
    #start_time = datetime.strptime(opt.cfg["start_date"], "%Y-%m-%d %H:%M:%S")
    #end_time = datetime.strptime(opt.cfg["end_date"], "%Y-%m-%d %H:%M:%S")

    if opt.calibration_start_date is None or opt.calibration_end_date is None:
        start_time, end_time = cal_targets.index[0], cal_targets.index[-1]
        logger.info(f"start_date and end_date not found in config; using calibration target data datetime index: {start_time} to {end_time}.")

    else:
        start_time = opt.calibration_start_date
        end_time = opt.calibration_end_date
        logger.info(f"Using start_date and end_date from config: {start_time} to {end_time}.")


    set_simulation_datetime(
        model=cal_model, 
        start_time=start_time, 
        end_time=end_time
    ).inp.save()

    counter = OptCount()
    
    success = de_calibration(
        cal_forcings=cal_forcings,
        cal_targets=cal_targets,
        in_model=cal_model,
        cal_params=cal_params,
        opt_config=opt,
        counter=counter,
    )

    #cal_model.inp.save()

    #path_prm = Path(os.path.join("output", 'calib_prm_' + routine + '.txt'))
    #path_res = Path(os.path.join("output", 'calib_res_' + routine + '.txt'))

    #save_dict(filename=path_prm, d=params)
    #save_dict(filename=path_res, d=iterations)

    logger.info("Calibration loop complete.")    


    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


    """reload and run the calibrated model"""
    #model = Model(get_model_path(cal_model))
    #run_swmm(model)
    #model.inp.save()

    """save a copy to the run directory (for archiving purposes)"""
    #model.inp.save(str(opt.run_dir / "calibrated_model.inp"))

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

        limits = np.array([cp.lower_limit,cp.upper_limit])

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
        new_val = truncate_values(new_val, limits)

        old_val = getattr(model.inp, cp.section).loc[idx,cp.attribute].values[0]
        getattr(model.inp, cp.section).loc[idx,cp.attribute] = new_val

        changes.loc[cp.tag,"cal_value"] = new_val
        changes.loc[cp.tag,"initial_value"] = old_val

    return model, changes


def truncate_values(values, limits):
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
    values[values < limits[0]] = limits[0]
    values[values > limits[1]] = limits[1]
    return values




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
        cal_params,
        cal_targets,
        counter,
        opt_config):
    """
    Optimization score function.

    :param values: Parameter values.
    :type values: list of floats
    :param in_model: Initial model.
    :type in_model: swmmio.Model object
    :param cal_params: Calibration parameters.
    :type cal_params: list[CalParam]
    :param cal_targets: Calibration targets.
    :type cal_targets: pd.DataFrame
    :param eval_nodes: Nodes to evaluate.
    :type eval_nodes: list of str
    :param counter: Optimization counter.
    :type counter: OptCount
    :param opt_config: Optimization configuration.
    :type opt_config: OptConfig
    :returns: Optimization score.
    :rtype: float
    """
    
    # Set up simulation preconfig with parameter updates
    cal_params.set_values(values)
    spc = cal_params.make_simulation_preconfig(model=opt_config.model)

    # Fix model strings to avoid blank timeseries bug
    #fix_model_strings(in_model)

    # Run simulation with updated parameters
    outputfile = str(Path(in_model).with_suffix('.out').resolve())
    
    with Simulation(str(in_model), outputfile=outputfile, sim_preconfig=spc) as sim:
        sim.execute()   

    # Evaluate model performance
    score_df, timeseries_results = eval_model(outputfile=outputfile, cal_targets=cal_targets, opt_config=opt_config)
    
    iter = counter.get_count()

    # Log results to files
    _log_results(score_df, cal_params, iter, opt_config, timeseries_results)
    
    counter.increment()

    return score_df[opt_config.score_function].mean()


def _log_results(score_df, cal_params, iter, opt_config, timeseries_results):
    """
    Log optimization results to files.
    
    :param score_df: Score dataframe.
    :type score_df: pd.DataFrame
    :param iter: Current iteration.
    :type iter: int
    :param opt_config: Optimization configuration.
    :type opt_config: OptConfig
    :param timeseries_results: Timeseries results.
    :type timeseries_results: dict
    """
    # Make a copy of the score df for writing to the results file
    score_df_copy = score_df.copy()
    # If NSE, flip the sign back (we flipped it in eval_model since we can only minimize)
    for col in score_df_copy.columns:
        if col in MAXIMIZE_FUNCTIONS:
            score_df_copy[col] = score_df_copy[col].apply(lambda x: -x)

    mi = score_df_copy.index.to_list()

    # Log scores
    for m in mi:
        now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
        line = f"{now},{iter},{m[0]},{m[1]},{opt_config.score_function[0]},{score_df_copy.loc[m, opt_config.score_function[0]]}\n"
        with open(opt_config.results_file_scores, 'a+') as f:
            f.write(line)

    # Log parameters if needed
    if np.mod(iter, opt_config.log_every_n) == 0:
        for ii, _ in enumerate(cal_params):
            now = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            line = f"{now},{iter},{cal_params[ii].ii},{cal_params[ii].initial_value},{cal_params[ii].opt_value},{cal_params[ii].opt_value_absolute}\n"
            with open(opt_config.results_file_params, 'a+') as f:
                f.write(line)

        # Save timeseries if requested
        if opt_config.save_timeseries:
            if not (opt_config.run_dir / "timeseries").exists():
                os.mkdir(opt_config.run_dir / "timeseries")

            for ts in timeseries_results.keys():
                filename = opt_config.run_dir / "timeseries" / f"{ts}_{iter}.pkl"
                timeseries_results[ts].to_pickle(filename)
    
    # Clean up, we will not need this model any more

    #fname_root = str(cal_model_tmp).split(".")[0]
    
    #os.remove(cal_model_tmp)
    #os.remove(Path(fname_root + ".out"))
    #os.remove(Path(fname_root + ".rpt"))
    return score_df[opt_config.score_function].mean()


def get_simulation_results(outputfile:str|Path, station_ids:list[str], params:list[str]):
        """
        Extract simulation results from SWMM output file for given stations and parameters.

        :param outputfile: Path to SWMM output file.
        :type outputfile: str or Path
        :param station_ids: List of station IDs.
        :type station_ids: list
        :param params: List of parameter names.
        :type params: list
        :returns: DataFrame with simulation results.
        :rtype: pd.DataFrame
        """
        dfs = []
        with Output(outputfile) as out:
            for station_id in station_ids:
                for param in params:  
                    if param in ["discharge(cms)", "flow(cms)"]:
                        res = out.node_series(station_id, shared_enum.NodeAttribute.TOTAL_INFLOW)
                    elif param in ["stage(m)", "wl(m)"]:
                        res = out.node_series(station_id, shared_enum.NodeAttribute.INVERT_DEPTH)
                    else:
                        raise ValueError(f"Parameter {param} not recognized")
                    dfs.append(
                        pd.DataFrame(
                            index=res.keys(),
                            data=res.values(),
                            columns=pd.MultiIndex.from_tuples([(param, station_id)])
                        ).copy()
                    )
        df = pd.concat(dfs, axis=1)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df



def eval_model(outputfile, cal_targets, opt_config):
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

    PERFORMANCE_FUNCTIONS = [func for func in dir(pf) if callable(getattr(pf, func)) and not func.startswith("_")]
    missing_funcs = [func for func in opt_config.score_function if func not in PERFORMANCE_FUNCTIONS]
    if len(missing_funcs) > 0:
        raise ValueError(f"Performance function {missing_funcs} not recognized")
    

    if len(opt_config.score_function) > 1:
        raise NotImplementedError("Multiple performance functions not implemented")
    else:
        score_function = opt_config.score_function[0]
        # TODO: add support for multiple performance functions
    score_fun = getattr(pf,score_function)

    station_ids = np.unique(cal_targets.columns.get_level_values(1))
    params = np.unique(cal_targets.columns.get_level_values(0))

    
    params = [p for p in params if p in opt_config.target_variables]
    if len(params) == 0:
        raise ValueError(f"No parameters to evaluate; target_variables: {opt_config.target_variables}, target_data: {cal_targets.columns.get_level_values(0)}")
    
    obs = cal_targets.loc[:, params]

    scores = pd.DataFrame(index=station_ids, columns=params)

    df = get_simulation_results(outputfile, station_ids, params)

    sim = df.copy()

    sim, obs = sync_timeseries(sim, obs)

    #print(obs.head())
    #sim = get_predictions_at_nodes(model=Model(cal_model_tmp), nodes=eval_nodes, param="FLOW_RATE")
    #sim = get_node_timeseries(model=Model(cal_model_tmp),nodes=eval_nodes, params=["TOTAL_INFLOW"])["TOTAL_INFLOW"][eval_nodes]
    #sim = sim.resample('15min').mean()

    if opt_config.normalize:
        scaler = get_scaler(obs)
        scaler = [s.values for s in scaler]
        obs = normalise(obs, scaler)
        sim = normalise(sim, scaler)

    #obs = obs.iloc[opt_config["warmup_length"]:, :]
    #sim = sim.iloc[opt_config["warmup_length"]:, :]

    score = [score_fun(obs.loc[:, col], sim.loc[:, col]) for col in obs.columns]

    scores = pd.DataFrame(index=obs.columns, data=score, columns=[score_function])


    if opt_config.normalize:
        obs = denormalise(obs, scaler)
        sim = denormalise(sim, scaler)

    

        #timeseries_results["hrt"].update({col: {"obs": obs.loc[:, col], "sim": sim.loc[:, col]} for col in obs.columns})


    # invert sign of NSE since opt function will always minimize
    if score_function in MAXIMIZE_FUNCTIONS:
        scores = scores.apply(lambda x: -x)

    return scores, {"obs": obs, "sim": sim}


def de_calibration(in_model,
                   cal_forcings,
                   cal_targets,
                   cal_params, 
                   opt_config,
                   counter:OptCount):
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
                                     counter=counter,
                                     opt_config=opt_config)
    


    bounds = cal_params.get_bounds()



    opt_args = opt_config.algorithm_options
    if opt_config.algorithm == "differential-evolution":
        opt_results = differential_evolution(func=opt_fun, bounds=bounds, **opt_args)

    elif opt_config.algorithm in ["Nelder-Mead","Powell","CG","BFGS","L-BFGS-B","TNC","COBYLA","SLSQP","trust-constr","dogleg","trust-ncg","trust-exact","trust-krylov"]:

        opt_results = minimize(method=opt_config.algorithm,fun=opt_fun, bounds=bounds, x0=[c.x0 for c in cal_params], options=opt_args)
    else:
        raise NotImplementedError(f"Algorithm {opt_config.algorithm} not implemented")
    
    #model = Model(in_model)
    #model, changes = set_params(cal_vals=opt_results.x, cal_params=cal_params, model=model)

    #model = fix_model_strings(model)
    #model.inp.save()

    #score = opt_results.fun
    #param_results = {c.tag:x for c, x in zip(cal_params, opt_results.x)}

    # clean up the temporary calibration SWMM files (NOT CURRENTLY USING TEMP DIR BECAUSE SWMMIO NOT HANDLING ABSOLUTE PATHS, MEANS NEED TO COPY PRECIP, HOTSTART, ETC. ON EACH ITERATION)
    """
    for file in [Path(in_model), Path(in_model).with_suffix('.out'), Path(in_model).with_suffix('.rpt')]:
        if file.exists():
            os.remove(file)
    """
    
    return True


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




