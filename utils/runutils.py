import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from utils.functions import load_dict



def initialize_run(run_loc: Path, name: str):
    """
    Creates a timestamped run directory to track calibration results.

    :param run_loc: Path to the directory where the run directory will be created.
    :type run_loc: Path
    :param name: Name of the calibration routine.
    :type name: str
    :returns: Path to the created run directory.
    :rtype: Path
    """
    now = datetime.now()
    current_time = now.strftime("%d-%m-%y-%H%M%S")
    run_dir = "run-{}_{}".format(name, current_time)
    x = Path(os.path.join(run_loc, run_dir))
    os.mkdir(x)
    
    return x

def summarize_runs(runs_dir:Path) -> list[Path]:
    run_dirs = [dir for dir in runs_dir.iterdir() if dir.is_dir()]
    run_dates = [datetime.strptime(d.name.split("_")[1], "%d-%m-%y-%H%M%S") for d in run_dirs]
    df = pd.DataFrame(index=run_dirs, data=run_dates, columns=["dates"])

    for d in run_dirs:
        cfg = load_dict(d / "opt_config.yaml")
        for k, v in cfg.items():
            if v == []:
                v = ''
            if isinstance(v, list):
                v = ', '.join(v)
            if isinstance(v, dict):
                v = ', '.join([f"{k1}: {v1}" for k1, v1 in v.items()])
            df.loc[d, k] = v

        if (not (d/"results_scores.txt").exists()) | (not (d/"results_params.txt").exists()):
            df.loc[d, "status"] = "failed"
        elif(len(pd.read_csv(d / "results_scores.txt")) == 0) | (len(pd.read_csv(d / "results_params.txt")) == 0):
            df.loc[d, "status"] = "failed"
        elif not (d/"calibrated_model.inp").exists():
            df.loc[d, "status"] = "incomplete"
        else:
            df.loc[d, "status"] = "complete"

        df = df.sort_values(by="dates")
    return df


