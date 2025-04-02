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


sys.path.append(r'C:\Users\everett\Documents\GitHub\camus_to')

data_file = r"C:\Users\everett\Documents\GitHub\camus_to\data\clean\camus_to.nc"


model_dir = Path(r"C:\Users\everett\Documents\GitHub\camus_to\data\models\swmm")
station_ids = [f.stem for f in model_dir.iterdir()]

for station_id in station_ids[:2]:
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

    training_start_date = df.index[training_iis[0]]
    training_end_date = df.index[training_iis[-1]]
    if (training_iis[-1] - training_iis[0]) > 365:
        training_end_date = training_start_date + pd.Timedelta(days=365)


    testing_start_date = df.index[testing_iis[0]]
    testing_end_date = df.index[testing_iis[-1]]
    

    oc = OptConfig(
        model_file = cal_model_file,
        forcing_data_file = model_dir / station_id / "forcings.pkl",
        target_data_file = model_dir / station_id / "targets.pkl",
        run_folder = model_dir / station_id / "runs",
        log_every_n = 10,
        algorithm = "Powell",
        algorithm_options={"maxiter":100},
        score_function=["nse"],
        name="3mo",
        target_variables=["discharge(cms)"],
        save_timeseries=True,
        calibration_start_date = training_start_date,
        calibration_end_date = training_end_date,
        validation_start_date = testing_start_date,
        validation_end_date = testing_end_date
    )

    cps = CalParams()
    cps.append(CalParam(section='infiltration', attribute='CurveNum', lower=0.5, upper=1, lower_limit=0, upper_limit=100, distributed=False))
    cps.append(CalParam(section='subcatchments', attribute='PercImperv', lower=0.5, upper=1, lower_limit=0, upper_limit=100, distributed=False))
    cps.append(CalParam(section='subcatchments', attribute='Area', lower=0.2, upper=4, lower_limit=0, upper_limit=100000000, distributed=False))
    cps.append(CalParam(section='subcatchments', attribute='Width', lower=1, upper=5, lower_limit=0, upper_limit=10000000, distributed=False))
    #cps.append(CalParam(section='subcatchments', attribute='Slope', lower=1, upper=0.5, lower_limit=0, upper_limit=100, distributed=False))
    cps.append(CalParam(section='xsections', attribute='Geom1', lower=1, upper=1, lower_limit=0.1, upper_limit=5, distributed=True))

    #cps.append(CalParam(section='conduits', attribute='Roughness', lower=0.5, upper=0.5, lower_limit=0, upper_limit=1, distributed=True))
    cps = cps.distribute(model=Model(str(model_file)))
    cps = cps.get_initial_values(model=Model(str(model_file)))
    cps = cps.set_relative_bounds()

    calibrate(opt_config=oc, cal_params=cps)
