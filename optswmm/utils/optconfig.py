"""Class to handle the configuration file for the optimization process"""

import yaml
from typing import Optional
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from swmmio import Model

from optswmm.defs import ALGORITHMS
from optswmm.utils.standardization import _standardize_file, _validate_target_data
from optswmm.utils.runutils import initialize_run
from optswmm.utils.swmmutils import set_model_datetimes
from optswmm.defs.filenames import DEFAULT_SCORES_FILENAME, DEFAULT_CAL_PARAMS_FILENAME, DEFAULT_PARAMS_FILENAME, DEFAULT_MODEL_FILENAME


class OptConfig:
    """Class to handle the configuration file for the optimization process"""
    def __init__(
            self,
            name: str = '',
            config_file: Optional[Path] = None,
            model_file: Optional[Path] = None,
            forcing_data_file: Optional[Path] = None,
            target_data_file: Optional[Path] = None,
            run_folder: Optional[Path] = None,
            calibration_nodes: Optional[list[str]] = None,
            target_variables: Optional[list[str]] = None,
            ignore_first_n: int = 0,
            normalize: bool = True,
            score_function: str = "mse",
            algorithm: str = "differential-evolution",
            parallel: bool = False,
            hierarchical: bool = False,
            log_every_n: int = 10,
            algorithm_options: Optional[dict] = None,
            save_timeseries: bool = False,
            calibration_start_date: Optional[str] = None,
            calibration_end_date: Optional[str] = None,
            validation_start_date: Optional[str] = None,
            validation_end_date: Optional[str] = None,
            report_step: Optional[pd.Timestamp] = None,
            dry_step: Optional[pd.Timestamp] = None,
            wet_step: Optional[pd.Timestamp] = None,
            ):
        
        """initialize the configuration file"""
        if config_file is None:
            if (model_file is None) or (forcing_data_file is None) or (target_data_file is None) or (run_folder is None):
                raise ValueError("If config file not provided, 'model_file', 'forcing_data_file', 'target_data_file', and 'run_folder' must be specified")
        
        self.name = name
        self.config_file = config_file
        self.model_file = model_file
        self.forcing_data_file = forcing_data_file
        self.target_data_file = target_data_file
        self.run_folder = run_folder
        self.run_dir = Path()
        self.calibration_nodes = calibration_nodes
        self.target_variables = target_variables
        self.ignore_first_n = ignore_first_n
        self.normalize = normalize
        self.score_function = score_function
        self.algorithm = algorithm
        self.parallel = parallel
        self.hierarchical = hierarchical
        self.log_every_n = log_every_n
        self.algorithm_options = algorithm_options
        self.save_timeseries = save_timeseries
        self.calibration_start_date = calibration_start_date
        self.calibration_end_date = calibration_end_date
        self.validation_start_date = validation_start_date
        self.validation_end_date = validation_end_date
        self.report_step = report_step
        self.dry_step = dry_step
        self.wet_step = wet_step


        # if config file provided, load and assign, overwriting default values
        if config_file is not None:
            #raise NotImplementedError("Config file not implemented")
            #self._validate_config_file(config_file)
            self.load_config(config_file)


    def _initialize_run(self):

                #self._standardize_config()
        self._initialize_model()
        self._standardize_config()
        self.model = Model(str(self.model_file))
        self._validate_target_data()
        self._assign_default_opt_options()
        
        if self.run_folder is None:
            raise ValueError("Run folder must be specified")

        if not self.run_folder.exists():
            self.run_folder.mkdir()

        self.run_dir = initialize_run(self.run_folder, self.name)
        self.results_file_params = self.run_dir / DEFAULT_PARAMS_FILENAME
        self.results_file_scores = self.run_dir / DEFAULT_SCORES_FILENAME
        self.calibrated_model_file = self.run_dir / DEFAULT_MODEL_FILENAME


        if not self.results_file_scores.exists():
            with open(self.results_file_scores, 'a+') as f:
                f.write('datetime,iter,obj_param,node,fun,score\n')
        
        if not self.results_file_params.exists():
            with open(self.results_file_params, 'a+') as f:
                f.write('datetime,iter,ii,model_val,cal_val,physical_val\n')
            
    def load_config(self, config_file):
        """load the configuration file"""
        with open(config_file, 'r') as file:
            for key, value in yaml.safe_load(file).items():
                setattr(self, key, value)
        self._standardize_config()

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
        if self.model_file is None:
            raise ValueError("Model file must be specified")
        
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file {self.model_file} does not exist.")
        
        if self.model_file.suffix != ".inp":
            raise ValueError(f"Model file must have extension .inp, got {self.model_file.suffix}")

        for file in [self.forcing_data_file, self.target_data_file]:
            if file is None:
                raise ValueError("Forcing data file and target data file must be specified")
            
            if not file.exists():
                raise FileNotFoundError(f"Forcing data file {file} does not exist.")
            
            if file.suffix not in [".pkl", ".csv"]:
                raise ValueError(f"Forcing data file must have extension .pkl or .csv, got {file.suffix}")

        #if not Path(self.run_dir).exists():
        #    Path(self.run_dir).mkdir()

        # Ensure datetime fields are in datetime format
        datetime_fields = [
            "calibration_start_date",
            "calibration_end_date",
            "validation_start_date",
            "validation_end_date",
        ]
        for field in datetime_fields:
            value = getattr(self, field)
            if value is not None and not isinstance(value, pd.Timestamp):
                try:
                    setattr(self, field, pd.Timestamp(value))
                except Exception as e:
                    raise ValueError(f"Invalid datetime format for {field}: {value}") from e

        if isinstance(self.score_function, str):
            self.score_function = [self.score_function]
        self.score_function = [x.lower() for x in self.score_function]

        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Algorithm {self.algorithm} not in {ALGORITHMS}")
        
        if self.name and "_" in self.name:
            warnings.warn("Tags should not contain underscores. Underscores will be replaced with hyphens.")
            self.name = self.name.replace("_", "-")

    def _initialize_model(self):
        """initialize the model object"""
        mdl = Model(str(self.model_file))
        if self.calibration_start_date is not None:
            mdl = set_model_datetimes(
                model=mdl,
                start_datetime=self.calibration_start_date,
                end_datetime=self.calibration_end_date,
                report_step=self.report_step,
                dry_step=self.dry_step,
                wet_step=self.wet_step,)
            
            mdl.inp.save()

    def _validate_target_data(self):
        """validate that the target data is compatible with the model"""
        _validate_target_data(self.target_data_file, self.model)


    def save_config(self):
        """save the configuration file"""
        with open(self.run_dir / "config.yml", 'w+') as file:
            # Convert Path objects to strings before saving
            cfg = self.__dict__.copy()
            for key, value in cfg.items():
                if isinstance(value, Path):
                    cfg[key] = str(value)
                elif isinstance(value, list):
                    cfg[key] = [str(x) for x in value]
                elif isinstance(value, dict):
                    cfg[key] = {k: str(v) for k, v in value.items()}
                elif isinstance(value, bool):
                    cfg[key] = str(value).lower()
                elif isinstance(value, pd.Timestamp) or isinstance(value, np.datetime64):
                    cfg[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(value, Model):
                    cfg[key] = None  # Models cannot be serialized directly
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


