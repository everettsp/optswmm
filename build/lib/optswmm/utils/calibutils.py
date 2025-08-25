"""Main calibration routine for SWMM models."""

# Standard library imports - only include what's necessary
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Third-party imports - optimized for performance
import numpy as np
import pandas as pd
# Import networkx only where needed
# import networkx as nx 
from scipy.optimize import differential_evolution, minimize
# Only import tqdm when progress bars are used
# from tqdm import tqdm
# Import gc and psutil only when monitoring is active
import gc
import psutil

# SWMM-related imports
from swmmio import Model
from pyswmm import Simulation, Output, SimulationPreConfig
from swmm.toolkit import shared_enum

# Local imports
from optswmm.utils.swmmutils import get_model_path
from optswmm.utils import perfutils as pf
from optswmm.utils.functions import sync_timeseries, invert_dict, load_dict
# Import this function only when needed
# from optswmm.utils.networkutils import get_downstream_nodes
from optswmm.utils.calparams import CalParam, CalParams
from optswmm.utils.standardization import load_timeseries
from optswmm.utils.optconfig import OptConfig

# Add lazy imports for rarely used modules
def get_downstream_nodes(*args, **kwargs):
    """Lazy import for networkx-dependent function"""
    import networkx as nx
    from optswmm.utils.networkutils import get_downstream_nodes as _get_downstream_nodes
    return _get_downstream_nodes(*args, **kwargs)

# Configure warnings - suppress unneeded ones
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

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
        model=Model(str(opt.model_file)), 
        start_time=start_time, 
        end_time=end_time
    ).inp.save()

    counter = OptCount()


    if opt.hierarchical:

        #df = get_calibration_order(cal_params, cal_model)
        #df = df[(df["node"] == "node_18") | (df["node"] == "None")]

        #cal_params = get_calibration_order(cal_params, cal_model)

        # remove nodes that aren't included among 'comparison nodes'
        # this doesn't generalize super well, as there may be cases you'd still want to calibrate a node that isn't in the comparison nodes
        # however in our case, this only includes the dummy-outfalls, since we have obs everywhere
        #cal_params = [c for c in cal_params if c.node in opt.calibration_nodes]

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
            cal_param_subset = CalParams([c for c in cal_params if c.level == level])
            
            node_subset = np.unique([c.node for c in cal_param_subset]).tolist()

            downstream_nodes = []
            for node in node_subset:
                downstream_nodes.extend(get_downstream_nodes(Model(str(opt.model_file)).network, node))
                
            node_subset = list(set(downstream_nodes))


            #node_subset = [node for node in node_subset if node in opt.cfg["calibration_nodes"]]
            #if "all" in [c.node.lower() for c in cal_param_subset]:
            #    node_subset = opt.calibration_nodes

            if len(node_subset) == 0:
                pass
            else:

                # For multi-index columns, select columns where 'nodes' level matches cal_nodes

                #cal_targets_subset = cal_targets.loc[:, cal_targets.columns.get_level_values('nodes').isin(node_subset)]

                opt_results = de_calibration(
                    cal_forcings=cal_forcings,
                    cal_targets=cal_targets,
                    in_model=Model(str(opt.model_file)),
                    cal_params=cal_param_subset,
                    opt_config=opt,
                    counter=counter,
                    )
                

                # Get the optimized parameter values
                values = opt_results.x
                
                # run the model with the updated parameters to update the model
                cal_param_subset.set_values(values)
                spc = cal_param_subset.make_simulation_preconfig(model=Model(opt.model_file))
                sim = Simulation(str(opt.model_file), sim_preconfig=spc)
                sim.execute()

                # replace the uncalibrated model file with the modified model file                            
                modified_model_file = opt.model_file.parent / f"{opt.model_file.stem}_mod.inp"

                if opt.model_file.exists():
                    opt.model_file.unlink()

                shutil.copy(str(modified_model_file.resolve()), str(opt.model_file.resolve()))


                #cal_model.inp.save()

    else:
            
        opt_results = de_calibration(
            cal_forcings=cal_forcings,
            cal_targets=cal_targets,
            in_model=Model(str(opt.model_file)),
            cal_params=cal_params,
            opt_config=opt,
            counter=counter,
        )


        cal_params.set_values(values)
        spc = cal_params.make_simulation_preconfig(model=Model(opt.calibrated_model_file))
        sim = Simulation(str(opt.calibrated_model_file), sim_preconfig=spc)
        sim.execute()
                    
        modified_model_file = opt.calibrated_model_file.parent / f"{opt.calibrated_model_file.stem}_mod.inp"
        opt.calibrated_model_file.unlink()
        shutil.copy(str(modified_model_file.resolve()), str(opt.calibrated_model_file.resolve()))




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

        #if cp.change == 'multiplicative':
            #new_val = (1+cal_vals[ii]) * cp.initial_value
        #elif cp.change == 'direct':
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



def get_open_file_count():
    """Get the current number of open file descriptors for this process."""
    try:
        process = psutil.Process()
        return len(process.open_files())
    except Exception:
        return -1

def log_open_files():
    """Log currently open files for debugging."""
    try:
        process = psutil.Process()
        open_files = process.open_files()
        print(f"Open files count: {len(open_files)}")
        for f in open_files:
            print(f"  {f.path}")
    except Exception as e:
        print(f"Could not get open files: {e}")

def de_score_fun(
        values,
        in_model,
        cal_params,
        cal_targets,
        counter,
        opt_config):
    """
    Optimization score function.
    """
    
    # Monitor file handles at start
    initial_file_count = get_open_file_count()
    
    # Set up simulation preconfig with parameter updates
    cal_params.set_values(values)
    spc = cal_params.make_simulation_preconfig(model=opt_config.model)

    # Run simulation with updated parameters
    outputfile = str(Path(in_model).with_suffix('.out').resolve())
    
    def cleanup_files():
        """Clean up simulation files"""
        for ext in [".out", ".rpt"]:
            try:
                file_path = Path(outputfile).with_suffix(ext)
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

    # In your de_score_fun function, modify the simulation execution:
    sim_failed = False
    sim = None
    try:
        sim = Simulation(str(in_model), outputfile=outputfile, sim_preconfig=spc)
        
        # Redirect output to suppress command line messages
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                
                max_retries = 8  # Reduced retries to prevent file buildup
                for attempt in range(max_retries):
                    try:
                        sim.execute()
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            cleanup_files()
                            time.sleep(0.01)
                            continue
                        else:
                            sim_failed = True
    finally:
        # Ensure simulation object is properly closed
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass

    if sim_failed:
        cleanup_files()
        return np.nan

    try:
        # Evaluate model performance
        score_df, timeseries_results = eval_model(outputfile=outputfile, cal_targets=cal_targets, opt_config=opt_config)
    except Exception as e:
        cleanup_files()
        return np.nan
    finally:
        # Clean up output files after evaluation
        cleanup_files()
        
        # Force garbage collection to close any lingering file handles
        gc.collect()
        
        # Monitor file handles at end and warn if increasing
        final_file_count = get_open_file_count()
        if final_file_count > initial_file_count + 5:  # Allow some tolerance
            print(f"Warning: File handle count increased from {initial_file_count} to {final_file_count}")
            if final_file_count > 100:  # Arbitrary threshold
                log_open_files()
    
    iter = counter.get_count()

    # Log results to files
    _log_results(score_df, cal_params, iter, opt_config, timeseries_results)
    
    counter.increment()

    # Handle NaN values by returning np.nanmean instead of mean
    score = score_df[opt_config.score_function].values
    score = score.astype(float)  # Ensure numeric
    return np.nanmean(score)


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
            line = f"{now},{iter},{cal_params[ii].ii},{cal_params[ii].initial_value},{cal_params[ii].opt_value},{cal_params[ii].model_value}\n"
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
        out = None
        try:
            out = Output(outputfile)
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
        finally:
            # Ensure output file is properly closed
            if out is not None:
                try:
                    out.close()
                except Exception:
                    pass
        
        if not dfs:
            raise ValueError("No data extracted from simulation results")
            
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


def de_calibration(
    in_model,
    cal_forcings,
    cal_targets,
    cal_params, 
    opt_config,
    counter: OptCount
):
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
        in_model = get_model_path(in_model, 'inp')

    opt_fun = lambda x: de_score_fun(
        x,
        in_model=in_model,
        cal_params=cal_params,
        cal_targets=cal_targets,
        counter=counter,
        opt_config=opt_config
    )

    bounds = cal_params.get_bounds()
    opt_args = opt_config.algorithm_options

    if opt_config.algorithm == "differential-evolution":
        opt_results = differential_evolution(
            func=opt_fun,
            bounds=bounds,
            **opt_args
        )
    elif opt_config.algorithm in [
        "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC",
        "COBYLA", "SLSQP", "trust-constr", "dogleg", "trust-ncg",
        "trust-exact", "trust-krylov"
    ]:
        opt_results = minimize(
            method=opt_config.algorithm,
            fun=opt_fun,
            bounds=bounds,
            x0=[c.x0 for c in cal_params],
            options=opt_args
        )
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
    
    return opt_results

