import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))  # add optswmm root to path for imports

from optswmm.utils.calparams import CalParam, CalParams
from optswmm.utils.calibutils import calibrate
from optswmm.utils.optconfig import OptConfig
from swmmio import Model

# Define paths relative to the optswmm root directory
optswmm_root = Path.cwd()  # Assuming we're in examples folder
test_dir = optswmm_root / "tests" / "fixtures"

# Model file (you'll need to ensure this exists in your test fixtures)
model_file = test_dir / "sample_models" / "Example1.inp"

# Data files
forcing_data = test_dir / "sample_data" / "forcing_data.csv"

# Configuration file
config_file = test_dir / "configs" / "sample_optconfig.yaml"
target_data_file = test_dir / "sample_data" / "example1_node_inflow.csv"

print(f"Model file: {model_file}")
print(f"Config file: {config_file}")
print(f"Test directory: {test_dir}")

from pyswmm import Simulation, Output
from swmm.toolkit import shared_enum

from swmmio import Model

# Create optimization configuration using Gaussian (Bayesian) optimizer
config_data = {
    'run_folder': str(Path("examples/calibration/runs").resolve()),
    'model_file': str(model_file.resolve()),
    'target_data_file': str(target_data_file.resolve()),
    'algorithm': 'Differential-Evolution',
    'algorithm_options': {
        'maxiter': 1000,
    },
    'score_function': ['kge'],
    'target_variables': ['flow(cms)'],
    'hierarchical': False,
    'normalize': False,
    'warmup_length': 0,
    'save_timeseries': True,
    'log_every_n': 1,
}

# Save configuration to file
config_file.parent.mkdir(parents=True, exist_ok=True)
import yaml

with open(config_file, 'w') as f:
    yaml.dump(config_data, f, default_flow_style=False)

# Load optimization configuration
opt_config = OptConfig(config_file=config_file)


# Create calibration parameters
cal_params = CalParams([
    # Subcatchment parameters
    CalParam(
        section='subcatchments',
        attribute='PercImperv',
        lower_abs=0,  # 90% decrease
        upper_abs=100,   # 200% increase
        lower_rel=-0.99,
        upper_rel=10,
        lower_limit_abs=0,
        upper_limit_abs=100,
        distributed=False,
        mode="linear",
        # mode="direct",  # Remove if not supported by CalParam
    ),
])


model = Model(str(model_file))
outfall_node = model.inp.outfalls.index[0]

model_vals = [80]
for ii in range(len(cal_params)):
    cal_params[ii].model_value = model_vals[ii]

cal_params = cal_params.preprocess(opt=opt_config)

spc = cal_params.make_simulation_preconfig(model=model)

print(f"Created {len(cal_params)} calibration parameters:")
for i, cp in enumerate(cal_params):
    print(f"  {i+1}. {cp.section}.{cp.attribute} (distributed: {cp.distributed})")

output_file = str(model_file.parent.resolve() / "output.rpt")
with Simulation(str(model_file), sim_preconfig=spc, outputfile=output_file) as sim:
    sim.execute()


node_ids = model.nodes().index.tolist()

# Collect node time series for TOTAL_INFLOW and save as a multi-column CSV (variable, node)
with Output(output_file) as out:
    # read series for each node
    node_series_map = {}
    for nid in node_ids:
        series = out.node_series(nid, shared_enum.NodeAttribute.TOTAL_INFLOW)
        if series:
            s = pd.Series(series).sort_index()
            s.index = pd.to_datetime(s.index)
            node_series_map[nid] = s

# create MultiIndex columns (variable, node) and combine into DataFrame
if node_series_map:
    # concat aligns on the union of timestamps
    df_nodes = pd.concat(node_series_map, axis=1)  # columns are node ids
    # make multiindex columns with top-level 'flow'
    cols = pd.MultiIndex.from_product([["flow(cms)"], list(df_nodes.columns)], names=["variable", "node"])
    df_nodes.columns = cols

    # ensure output directory exists and save
    target_data_file.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(target_data_file)
    print(f"Saved node output (TOTAL_INFLOW) for {len(node_series_map)} nodes to: {target_data_file}")
else:
    print("No node series were retrieved.")


calibrate(opt_config=opt_config, cal_params=cal_params)
