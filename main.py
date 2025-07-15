import sys
from pathlib import Path
from datetime import datetime
import shutil
import xarray as xr
import numpy as np
import pandas as pd

from pyswmm import Simulation, Nodes, Links, Subcatchments, SystemStats, Output
from swmm.toolkit import shared_enum
from swmmio import Model

from utils.calibutils import calibrate
from utils.swmmutils import set_model_datetimes
from utils.calparams import CalParam, CalParams
from utils.optconfig import OptConfig
import utils.perfutils as pf
import warnings
from tqdm import tqdm

sys.path.append(r'C:\Users\everett\Documents\GitHub\camus_to')
data_file = r"C:\Users\everett\Documents\GitHub\camus_to\data\clean\camus_to.nc"
model_dir = Path(r"C:\Users\everett\Documents\GitHub\camus_to\data\models\swmm")
station_ids = [f.stem for f in model_dir.iterdir()]


for station_id in tqdm(station_ids):
    model_file = model_dir / station_id / f"{station_id}.inp"

    model_dir = Path(r"C:\Users\everett\Documents\GitHub\camus_to\data\models\swmm")
    model_file = model_dir / station_id / f"{station_id}.inp"

    # Copy the model file to a new location

    cal_model_file = model_file.with_name(model_file.stem + "_cal" + model_file.suffix)

    shutil.copy(model_file, cal_model_file)

    mdl = Model(str(cal_model_file))


    model_file = model_dir / station_id / f"{station_id}.inp"

    forcings_df = pd.read_pickle(model_dir / station_id / "forcings.pkl")
    targets_df = pd.read_pickle(model_dir / station_id / "targets.pkl")
    df = pd.merge(forcings_df, targets_df, left_index=True, right_index=True)

    non_nan_discharge_iis = np.argwhere(~df["discharge(cms)"].isna())
    non_nan_discharge_iis = non_nan_discharge_iis[:,0]
    midpoint = len(non_nan_discharge_iis) // 2
    training_iis = non_nan_discharge_iis[:midpoint]
    testing_iis = non_nan_discharge_iis[midpoint:]

    max_train_len_days = 365
    if len(training_iis) < max_train_len_days:
        warnings.warn(f"Training data for {station_id} is less than {max_train_len_days} days. Skipping calibration.")
        continue

    training_start_date = df.index[training_iis[0]]
    training_end_date = df.index[training_iis[-1]]
    if (training_iis[-1] - training_iis[0]) > max_train_len_days:
        training_end_date = training_start_date + pd.Timedelta(days=max_train_len_days)

    testing_start_date = df.index[testing_iis[0]]
    testing_end_date = df.index[testing_iis[-1]]
    
    oc = OptConfig(
        model_file = cal_model_file,
        forcing_data_file = model_dir / station_id / "forcings.pkl",
        target_data_file = model_dir / station_id / "targets.pkl",
        run_folder = model_dir / station_id / "runs",
        log_every_n = 1,
        algorithm = "Powell",
        algorithm_options={"maxiter":5, "ftol":0.01},
        score_function=["nse"],
        name="higher_res", # no underscores
        target_variables=["discharge(cms)"],
        save_timeseries=True,
        calibration_start_date = training_start_date,
        calibration_end_date = training_end_date,
        validation_start_date = testing_start_date,
        validation_end_date = testing_end_date,
        report_step = pd.Timestamp("01:00:00"),
        dry_step = pd.Timestamp("01:00:00"),
        wet_step = pd.Timestamp("00:30:00"),
    )

    cps = CalParams()
    cps.append(CalParam(section='infiltration', attribute='CurveNum', lower=-1, upper=1, lower_limit=0, upper_limit=100, distributed=False, relative=True))
    cps.append(CalParam(section='subcatchments', attribute='PercImperv', lower=-1, upper=1, lower_limit=0, upper_limit=100, distributed=False, relative = True))
    #cps.append(CalParam(section='subcatchments', attribute='Area', lower=-0.2, upper=0.2, lower_limit=0, upper_limit=100000000, distributed=False, relative=True))
    cps.append(CalParam(section='subcatchments', attribute='Width', lower=-0.999, upper=8, lower_limit=0, upper_limit=10000000, distributed=False, relative=True))
    cps.append(CalParam(section='subareas', attribute='N-Perv', lower=-0.99999999, upper=1, lower_limit=0, upper_limit=10000000, distributed=False, relative=True))
    cps.append(CalParam(section='subareas', attribute='N-Imperv', lower=-0.99999999, upper=1, lower_limit=0, upper_limit=10000000, distributed=False, relative=True))
    #cps.append(CalParam(section='subcatchments', attribute='Slope', lower=1, upper=0.5, lower_limit=0, upper_limit=100, distributed=False))
    cps.append(CalParam(section='xsections', attribute='Geom2', lower=-0.2, upper=0.2, lower_limit=0.05, upper_limit=5, distributed=False, relative=True))
    cps.append(CalParam(section='inflows', attribute='Baseline', lower=-0.2, upper=0.2, lower_limit=0, upper_limit=10, distributed=False, relative=True))

    #cps.append(CalParam(section='conduits', attribute='Roughness', lower=0.5, upper=0.5, lower_limit=0, upper_limit=1, distributed=True))
    cps = cps.distribute(model=Model(str(model_file)))
    cps = cps.get_initial_values(model=Model(str(model_file)))
    cps = cps.relative_bounds_to_absolute()

    # here we skip calibrations with the same name, to deal with the script failing often

    if (not oc.run_folder.exists()) or (~np.any([oc.name in f.stem for f in oc.run_folder.iterdir()])):
        try:
            print(f"Calibrating {station_id}...")
            calibrate(opt_config=oc, cal_params=cps)
        except Exception as e:
            print(f"An error occurred during calibration: {e}") 