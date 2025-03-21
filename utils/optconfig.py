"""Class to handle the configuration file for the optimization process"""

import yaml
from defs import ALGORITHMS, CALIBRATION_ROUTINES, CONFIG_OPTIONS
from pathlib import Path
import warnings


class OptConfig:
    """Class to handle the configuration file for the optimization process"""
    def __init__(self, config_path):
        """initialize the configuration file"""
        self.base_model_dir:str|Path = None
        self.calibration_dir:str|Path = None
        self.calibration_nodes:list[str] = None
        self.target_variables:list[str] = None
        self.ignore_first_n:int = None
        self.normalize:bool = True
        self.score_function:str = None
        self.routine:str = None
        self.algorithm:str = None
        self.parallel:bool = False
        self.hierarchial:bool = False
        self.log_every_n:int = None
        #self._standardize_config()

    def load_config(self):
        """load the configuration file"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def _standardize_config(self):
        for key, value in self.cfg.items():
            if key in CONFIG_OPTIONS:
                if not isinstance(value, CONFIG_OPTIONS[key]):
                    raise ValueError(f"Value {value} for key {key} is not of type {CONFIG_OPTIONS[key]}")

    def save_config(self, output_path):
        """save the configuration file"""
        cfg = self.cfg.copy()
        with open(output_path, 'w') as file:
            # Convert Path objects to strings before saving
            for key, value in cfg.items():
                if isinstance(value, Path):
                    cfg[key] = str(value)

            yaml.safe_dump(cfg, file)

    def _standardize_config(self):
        """standardize the configuration file"""

        if self.cfg["algorithm"] not in ALGORITHMS:
            raise ValueError(f"Algorithm {self.cfg['algorithm']} not in {ALGORITHMS}")
        
        if "calibration_nodes" not in list(self.cfg.keys()):
            Warning("No calibration nodes specified; using all nodes in load_calibration_data")
            self.cfg["calibration_nodes"] = []

        #if calibration_nodes is 'all', set to empty list

        if [x.lower() for x in self.cfg["calibration_nodes"]] == ["all"]:
            self.cfg["calibration_nodes"] = []
        
        if type(self.cfg["calibration_nodes"]) is list:
            pass
        elif type(self.cfg["calibration_nodes"]) is str:
            self.cfg["calibration_nodes"] = [self.cfg["calibration_nodes"]]
        else:
            raise ValueError("calibration_nodes must be a list or string")

        if "target_variables" not in list(self.cfg.keys()):
            warnings.warn("No target variables specified; using all variables in load_calibration_data")
            self.cfg["target_variables"] = {}

        """checks if the routine is implemented"""
        if self.cfg["routine"] not in CALIBRATION_ROUTINES:
            raise NotImplementedError(f"Routine {self.cfg['routine']} is not implemented, choices include: {CALIBRATION_ROUTINES}")

        if isinstance(self.cfg["base_model_dir"], str):
            self.cfg["base_model_dir"] = Path(self.cfg["base_model_dir"])

        if self.cfg["calibration_dir"] == "":
            self.cfg["calibration_dir"] = self.cfg["base_model_dir"].parent / f"{self.cfg['routine']}_calibration"
            warnings.warn(f"Calibration directory not found, using default: {self.cfg['calibration_dir']}")

        if isinstance(self.cfg["calibration_dir"], str):
            self.cfg["calibration_dir"] = Path(self.cfg["calibration_dir"])

        if not isinstance(self.cfg["warmup_length"], int):
            self.warmup_length = int(self.warmup_length)
        
        
        if self.cfg["score_function"] not in ["nse","mse"]:
            raise ValueError(f"Score function {self.cfg['score_function']} not in ['nse','mse']")
        return self


