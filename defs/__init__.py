# Standard library imports
import os
import yaml
from yaml import SafeLoader
from pathlib import Path

_DEFS_PATH = os.path.abspath(os.path.dirname(__file__))

def load_yaml(filename:Path) -> dict:
    """load project config file containing data dirs"""
    with open(filename) as f:
        return yaml.load(f, Loader=SafeLoader)

dirs = load_yaml(Path(_DEFS_PATH) / 'config.yaml')


dir_root = os.path.abspath(os.path.dirname(__name__))
data_dir = Path(os.path.join(dir_root, 'data'))
# DC: data_dir = Path(dirs['data_dir'])
# DC: iw_model_dir = data_dir / 'InfoWorks'
iw_model_dir = os.path.join(data_dir, 'InfoWorks')


# many to one mapping for different conduit abbreviations
CALIBRATION_ROUTINES = ["dry", "wet", "tss"]
CALIBRATION_TARGETS = ["flow","hrt","depth","vol","tss"]
CALIBRATION_FORCINGS = ["precip","temp"]

SWMM_SECTION_SUBTYPES = {
    "infiltration":"subcatchments",
    "subareas":"subcatchments",
    "subcatchments":"subcatchments",
    "conduits":"links",
    "xsections":"links",
    "dwf":"nodes",
    "inflows":"nodes",
    "pollutants":"landuses",
    "buildup":"landuses",
    "washoff":"landuses",
    "pollutants":"landuses",
    "buildup":"landuses",
    "washoff":"landuses",
    "rdii":"node",
    "hydrographs":"hydrographs",
    }


ALGORITHMS = ["differential-evolution","Nelder-Mead","Powell","CG","BFGS","L-BFGS-B","TNC","COBYLA","SLSQP","trust-constr","dogleg","trust-ncg","trust-exact","trust-krylov"]

SWMM_TIME_FMT = "%Y-%m-%d %H:%M:%S"

LOG_TIME_FMT = "%d-%m-%y-%H%M%S"

CONFIG_OPTIONS = {
    "log_every_n":int,
    "start_date":str,
    "end_date":str,
    "do_parallel":bool,
    "hierarchial":bool,
    "calibration_nodes":list,
    "warmup_length":int,
    "normalize":bool,
    "score_function":str,
    "target_variables":dict|list,
    "algorithm":str,
    "diffevol_maxiter":int,
    "diffevol_popsize":int,
    "diffevol_workers":int,
    "diffevol_mutation":float,
    "diffevol_seed":int,
    "minimize_maxiter":int,
    "minimize_xatol":float,
    "minimize_disp":bool,
    "minimize_adaptive":bool
}