import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from optswmm.utils.functions import load_dict
import plotly.graph_objs as go

from optswmm.defs.filenames import DEFAULT_SCORES_FILENAME, DEFAULT_CAL_PARAMS_FILENAME, DEFAULT_PARAMS_FILENAME, DEFAULT_MODEL_FILENAME


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

    runs = [OptRun(run_loc=run_dir) for run_dir in run_dirs]

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



        if not (d / DEFAULT_SCORES_FILENAME).exists():
            df.loc[d, "max_iter"] = 0
        else:
            scores = pd.read_csv(d / DEFAULT_SCORES_FILENAME).rename(columns=lambda x: x.strip())
            df.loc[d, "max_iter"] = len(scores)
            df.loc[d, "best_score"] = scores["score"].min()

        cal_params_file = d / DEFAULT_CAL_PARAMS_FILENAME
        params_fesults_file = d/ DEFAULT_PARAMS_FILENAME

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



def plot_run_minimization(df:Path|str|pd.DataFrame):
    """
    Plot the minimization scores from a DataFrame or directory of runs.
    :param df: DataFrame containing run scores or path to a directory with run results.
    :type df: Path, str, or pd.DataFrame
    :raises ValueError: If the DataFrame does not contain 'score' and 'iter' columns or is empty.
    """

    if not all(col in df.columns for col in ["score", "iter"]):
        raise ValueError("DataFrame must contain 'score' and 'iter' columns.")

    if df.empty:
        raise ValueError("DataFrame is empty. No runs to plot.")

    trace = go.Scatter(
        x=df["iter"],
        y=df["score"],
        mode='lines',
        name="Run Scores"
    )
    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_layout(
        title="Run Scores",
        xaxis_title="Iteration",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend_title="Run"
    )
    fig.show()
    

def plot_runs_minimization(runs):
    """
    Plot the minimization scores from a DataFrame or directory of runs.

    :param df: DataFrame containing run scores or path to a directory with run results.
    :type df: Path, str, or pd.DataFrame
    :raises ValueError: If the DataFrame does not contain 'score' and 'iter' columns or is empty.
    """

    if isinstance(runs, str):
        runs = Path(runs)

    if isinstance(runs, Path):
        if runs.is_dir():
            runs = summarize_runs(runs)
        else:
            raise ValueError("Path must be a directory containing folders with individual runs.")

    if not all(col in runs.columns for col in ["score", "iter"]):
        raise ValueError("DataFrame must contain 'score' and 'iter' columns.")

    if runs.empty:
        raise ValueError("DataFrame is empty. No runs to plot.")

    for run_dir in runs.index:
        scores_file = run_dir / DEFAULT_SCORES_FILENAME
        scores_df = pd.read_csv(scores_file, sep=",", header=0, index_col=0).rename(columns=lambda x: x.strip())
        trace = go.Scatter(x=scores_df["iter"], y=scores_df["score"], mode='lines', name=run_dir.name)
        if 'fig' not in locals():
            fig = go.Figure()
        fig.add_trace(trace)

    fig.update_layout(
        title="Run Scores",
        xaxis_title="Iteration",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend_title="Run"
    )
    fig.show()

# Example usage:
# plot_run_scores(scores_df)
class OptRun:
    """
    Class to handle the optimization run, including initialization and plotting of results.
    """

    def __init__(self, run_loc: Path):
        self.run_dir = run_loc
        self.name = run_loc.name
        try:
            self.date = datetime.strptime(self.name.split("_")[1], "%d-%m-%y-%H%M%S")
        except Exception:
            self.date = None
        self.scores = self.get_scores()
        self.params = None

    def get_scores(self):
        """
        Get the scores from the optimization run.
        """
        scores_file = self.run_dir / "results_scores.txt"
        if not scores_file.exists():
            raise FileNotFoundError(f"Scores file {scores_file} does not exist.")
        
        return pd.read_csv(scores_file, sep=",", header=0, index_col=0).rename(columns=lambda x: x.strip())

    def get_params(self):
        """
        Get the parameters from the optimization run.
        """
        params_file = self.run_dir / "results_params.csv"
        if not params_file.exists():
            return None
        params = pd.read_csv(params_file)
        if "datetime" in params.columns:
            params.set_index(pd.DatetimeIndex(params["datetime"]), inplace=True)
            params.drop(columns=["datetime"], inplace=True)
        self.params = params

        # load the params metadata to match the param id (ii) to the parameter name
        param_index = pd.read_csv(self.run_dir / "calibration_parameters.csv", index_col=0)
        param_index = param_index.set_index("ii")
        param_index["param_name"] = [f"{p['section']}_{p['attribute']}" for _, p in param_index.iterrows()]

        self.params["param_name"] = self.params["ii"].map(param_index["param_name"])

        return params


class OptRuns:
    """
    Iterable collection of OptRun objects.
    """

    def __init__(self, runs_dir: Path):
        if isinstance(runs_dir, str):
            runs_dir = Path(runs_dir)
        if not runs_dir.is_dir():
            raise ValueError("runs_dir must be a directory containing run folders.")
        self.run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        self.runs = [OptRun(run_loc=run_dir) for run_dir in self.run_dirs]

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, idx):
        return self.runs[idx]

    def __len__(self):
        return len(self.runs)
    
    def get_df(self):
        """
        Get a DataFrame summarizing the runs.
        """
        data = {
            "name": [run.name for run in self.runs],
            "date": [run.date for run in self.runs],
            "scores": [run.scores for run in self.runs],
            "params": [run.get_params() for run in self.runs]
        }
        return pd.DataFrame(data)
    
    def plot_scores(self):
        """
        Plot the scores of all runs in the collection.
        """
        if not self.runs:
            raise ValueError("No runs to plot.")
        
        fig = go.Figure()
        for run in self.runs:
            if run.scores is not None:
                trace = go.Scatter(
                    x=run.scores["iter"],
                    y=run.scores["score"],
                    mode='lines',
                    name=run.name
                )
                fig.add_trace(trace)

