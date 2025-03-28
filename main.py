import sys
from pathlib import Path
from datetime import datetime
import shutil
import xarray as xr

from pyswmm import Simulation, Nodes, Links, Subcatchments, SystemStats, Output
from swmm.toolkit import shared_enum
from swmmio import Model

from utils.calibutils import calibrate
from utils.swmmutils import set_model_datetimes
from utils.calparams import CalParam, CalParams
from utils.optconfig import OptConfig


sys.path.append(r'C:\Users\everett\Documents\GitHub\camus_to')

data_file = r"C:\Users\everett\Documents\GitHub\camus_to\data\clean\camus_to.nc"

station_id = "HY095"
model_dir = Path(r"C:\Users\everett\Documents\GitHub\camus_to\data\models\swmm")
model_file = model_dir / station_id / f"{station_id}.inp"

# Copy the model file to a new location

cal_model_file = model_file.with_name(model_file.stem + "_cal" + model_file.suffix)

shutil.copy(model_file, cal_model_file)

mdl = Model(str(cal_model_file))

start_time = datetime(2021, 1, 1, 0, 0, 0)
end_time = datetime(2021, 2, 1, 0, 0, 0)
mdl = set_model_datetimes(model=mdl, start_datetime=start_time, end_datetime=end_time)
mdl.inp.save()


oc = OptConfig(
    model_file = cal_model_file,
    forcing_data_file = model_dir / station_id / "forcings.pkl",
    target_data_file = model_dir / station_id / "targets.pkl",
    run_dir = model_dir / station_id / "runs",
    log_every_n = 1 ,
    algorithm = "Powell",
    algorithm_options={"eps":1e-1, "maxiter":1000},
    score_function="nse",
)

cps = CalParams()
cps.append(CalParam(section='infiltration', attribute='CurveNum', lower=0.5, upper=0.5, lower_limit=0, upper_limit=100, distributed=True))
cps.append(CalParam(section='subcatchments', attribute='PercImperv', lower=0.2, upper=0.2, lower_limit=0, upper_limit=100, distributed=True))
cps.append(CalParam(section='xsections', attribute='Geom1', lower=1, upper=1, lower_limit=0.1, upper_limit=5, distributed=True))
cps.append(CalParam(section='conduits', attribute='Roughness', lower=0.5, upper=0.5, lower_limit=0, upper_limit=1, distributed=True))
cps = cps.distribute(model=Model(str(model_file)))
cps.set_relative_bounds()

calibrate(opt_config=oc, cal_params=cps)
