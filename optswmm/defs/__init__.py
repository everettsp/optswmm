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


PARAM_INDICES = {
    "subcatchments": {
        "RainGage": 1,
        "Outlet": 2,
        "Area": 3,
        "Width": 4,
        "PercImperv": 5,
        "Slope": 6,
        "CurbLength": 7,
        "SnowPack": 8
    },
    "subareas": {
        "N-Imperv": 1,
        "N-Perv": 2,
        "S-Imperv": 3,
        "S-Perv": 4,
        "PctZero": 5,
        "RouteTo": 6
    },
    "infiltration": {
        "Param1": 1,
        "Param2": 2,
        "Param3": 3,

        "Param4": 4,
        "Param5": 5,
        "CurveNum": 1,
    },
    "junctions": {
        "Elevation": 1,
        "MaxDepth": 2,
        "InitDepth": 3,
        "SurDepth": 4,
        "Aponded": 5
    },
    "conduits":{
        "FromNode":1,
        "ToNode":2,
        "Length":3,
        "Roughness":4,
        "InOffset":5,
        "OutOffset":6,
        "InitFlow":7,
        "MaxFlow":8,
    },
    "xsections":{
        "Shape":1,
        "Geom1":2,
        "Geom2":3,
    },
    "inflows":{
        "Mfactor":4,
        "Sfactor":5,
        "Baseline":6,
        "Pattern":7,
    },
    "dwf":{
        "AverageValue": 2,
    },
    "hydrographs":{
        "Response":2,
        "R":3,
        "T":4,
        "K":5,
        "Dmax":6,
        "Drecov":7,
        "Dinit":8,
    },
    "rdii":{
        "SewerArea": 2,
    },
}

ROW_ATTRIBUTES = {"hydrographs":{"Response":2}}

ROW_INDICES = {"hydrographs":{
        "Short": 1,
        "Medium": 2,
        "Long": 3
    }
}


