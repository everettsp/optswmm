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

def summarize_runs(runs_dir:Path, drop_incomplte=True) -> list[Path]:
    run_dirs = [dir for dir in runs_dir.iterdir() if dir.is_dir()]
    run_dates = [datetime.strptime(d.name.split("_")[1], "%d-%m-%y-%H%M%S") for d in run_dirs]
    df = pd.DataFrame(index=run_dirs, data=run_dates, columns=["dates"])

    for d in run_dirs:
        cfg = load_dict(d / "config.yml")
        for k, v in cfg.items():
            if v == []:
                v = ''
            if isinstance(v, list):
                v = ', '.join(v)
            if isinstance(v, dict):
                v = ', '.join([f"{k1}: {v1}" for k1, v1 in v.items()])
            df.loc[d, k] = v



        if not (d/"results_scores.txt").exists:
            df.loc[d, "max_iter"] = 0
        else:
            scores = pd.read_csv(d/"results_scores.txt").rename(columns=lambda x: x.strip())
            df.loc[d, "max_iter"] = len(scores)
            df.loc[d, "best_score"] = scores["score"].min()
        
        cal_params_file = d / "calibration_parameters.csv"
        params_fesults_file = d/"results_params.txt"

        if (not cal_params_file.exists()) | (not params_fesults_file.exists()):
            df.loc[d, "max_iter"] = 0
            continue

        df_params = pd.read_csv(params_fesults_file).rename(columns=lambda x: x.strip())
        cal_params = pd.read_csv(cal_params_file)

        if "ii" not in df_params.columns:
            df.loc[d, "max_iter"] = 0

        if "ii" not in cal_params.columns:
            df.loc[d, "max_iter"] = 0
            
    df = df.sort_values(by="dates")
        
    if drop_incomplte:
        df = df[df["max_iter"] > 0]
    return df