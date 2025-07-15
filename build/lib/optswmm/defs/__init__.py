# Standard library imports
import os
import yaml
from yaml import SafeLoader
from pathlib import Path

_DEFS_PATH = os.path.abspath(os.path.dirname(__file__))

dir_root = os.path.abspath(os.path.dirname(__name__))
data_dir = Path(os.path.join(dir_root, 'data'))

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

SWMM_DATE_FMT = "%m/%d/%Y"
SWMM_TIME_FMT = "%H:%M:%S"
SWMM_DATETIME_FMT = SWMM_DATE_FMT + " " + SWMM_TIME_FMT

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



