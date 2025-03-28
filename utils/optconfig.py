"""Class to handle the configuration file for the optimization process"""

import yaml
from defs import ALGORITHMS, CALIBRATION_ROUTINES, CONFIG_OPTIONS
from pathlib import Path
import warnings
from swmmio import Model
from utils.standardization import _standardize_file, _validate_target_data

class OptConfig:
    """Class to handle the configuration file for the optimization process"""
    def __init__(
            self,
            config_file:Path=None,
            model_file:Path=None,
            forcing_data_file:Path=None,
            target_data_file:Path=None,
            run_dir:Path=None,
            calibration_nodes:list[str]=None,
            target_variables:list[str]=None,
            ignore_first_n:int=0,
            normalize:bool=True,
            score_function:str="mse",
            routine:str=None,
            algorithm:str="differential-evolution",
            parallel:bool=False,
            hierarchial:bool=False,
            log_every_n:int=10,
            algorithm_options:dict=None,
            ):
        
        """initialize the configuration file"""
        if config_file is None:
            if (model_file is None) or (forcing_data_file is None) or (target_data_file is None) or (run_dir is None):
                raise ValueError("If config file not provided, 'model_file', 'forcing_data_file', 'target_data_file', and 'run_dir' must be specified")
        
        self.config_file = config_file
        self.model_file = model_file
        self.forcing_data_file = forcing_data_file
        self.target_data_file = target_data_file
        self.run_dir = run_dir
        self.calibration_nodes = calibration_nodes
        self.target_variables = target_variables
        self.ignore_first_n = ignore_first_n
        self.normalize = normalize
        self.score_function = score_function
        self.routine = routine
        self.algorithm = algorithm
        self.parallel = parallel
        self.hierarchial = hierarchial
        self.log_every_n = log_every_n
        self.algorithm_options = algorithm_options
    
        # if config file provided, load and assign, overwriting default values
        if config_file is not None:
            raise NotImplementedError("Config file not implemented")

            self._validate_config_file(config_file)
            self.load_config(config_file)

        #self._standardize_config()

        self._standardize_config()
        self.model = Model(str(self.model_file))
        self._validate_target_data()
        self._assign_default_opt_options()

    def load_config(self, config_file):
        """load the configuration file"""
        with open(config_file, 'r') as file:
            for key, value in yaml.safe_load(file).items():
                setattr(self, key, value)


    def _assign_default_opt_options(self):
        if (self.algorithm == "differential-evolution") & (self.algorithm_options is None):
            self.algorithm_options = {"maxiter":1000, "popsize":50}
        elif (self.algorithm == "Nelder-Mead") & (self.algorithm_options is None):
            self.algorithm_options = {}
        elif (self.algorithm == "GA") & (self.algorithm_options is None):
            self.algorithm_options = {"maxiter":1000, "popsize":50}
        elif (self.algorithm == "SA") & (self.algorithm_options is None):
            self.algorithm_options = {"maxiter":1000, "popsize":50}
        elif self.algorithm_options is None:
            raise ValueError(f"Cannot find default optimization options for algorithm {self.algorithm}")
        
    def _standardize_config(self):
        _standardize_file(self.model_file, exists=True, ext=".inp")
        _standardize_file(self.forcing_data_file, exists=True, ext=".pkl")
        _standardize_file(self.target_data_file, exists=True, ext=".pkl")
        
        _standardize_file(self.run_dir, exists=False)
        if not Path(self.run_dir).exists():
            Path(self.run_dir).mkdir()

        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Algorithm {self.algorithm} not in {ALGORITHMS}")
        



    def _validate_target_data(self):
        """validate that the target data is compatible with the model"""
        _validate_target_data(self.target_data_file, self.model)

    def save_config(self, output_path):
        """save the configuration file"""
        with open(output_path, 'w') as file:
            # Convert Path objects to strings before saving
            cfg = self.__dict__.copy()
            for key, value in cfg.items():
                if isinstance(value, Path):
                    cfg[key] = str(value)

            yaml.safe_dump(cfg, file)

        # def _standardize_config(self):
        #     """standardize the configuration file"""

        #     if self.cfg["algorithm"] not in ALGORITHMS:
        #         raise ValueError(f"Algorithm {self.cfg['algorithm']} not in {ALGORITHMS}")
            
        #     if "calibration_nodes" not in list(self.cfg.keys()):
        #         Warning("No calibration nodes specified; using all nodes in load_calibration_data")
        #         self.cfg["calibration_nodes"] = []

        #     #if calibration_nodes is 'all', set to empty list

        #     if [x.lower() for x in self.cfg["calibration_nodes"]] == ["all"]:
        #         self.cfg["calibration_nodes"] = []
            
        #     if type(self.cfg["calibration_nodes"]) is list:
        #         pass
        #     elif type(self.cfg["calibration_nodes"]) is str:
        #         self.cfg["calibration_nodes"] = [self.cfg["calibration_nodes"]]
        #     else:
        #         raise ValueError("calibration_nodes must be a list or string")

        #     if "target_variables" not in list(self.cfg.keys()):
        #         warnings.warn("No target variables specified; using all variables in load_calibration_data")
        #         self.cfg["target_variables"] = {}

        #     """checks if the routine is implemented"""
        #     if self.cfg["routine"] not in CALIBRATION_ROUTINES:
        #         raise NotImplementedError(f"Routine {self.cfg['routine']} is not implemented, choices include: {CALIBRATION_ROUTINES}")

        #     if isinstance(self.cfg["base_model_dir"], str):
        #         self.cfg["base_model_dir"] = Path(self.cfg["base_model_dir"])

        #     if self.cfg["calibration_dir"] == "":
        #         self.cfg["calibration_dir"] = self.cfg["base_model_dir"].parent / f"{self.cfg['routine']}_calibration"
        #         warnings.warn(f"Calibration directory not found, using default: {self.cfg['calibration_dir']}")

        #     if isinstance(self.cfg["calibration_dir"], str):
        #         self.cfg["calibration_dir"] = Path(self.cfg["calibration_dir"])

        #     if not isinstance(self.cfg["warmup_length"], int):
        #         self.warmup_length = int(self.warmup_length)
            
            
        #     if self.cfg["score_function"] not in ["nse","mse"]:
        #         raise ValueError(f"Score function {self.cfg['score_function']} not in ['nse','mse']")
        #     return self


