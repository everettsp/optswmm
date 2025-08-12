import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from optswmm.utils.functions import load_dict
import plotly.graph_objs as go
import numpy as np
import warnings
from swmmio import Model

from optswmm.defs.filenames import DEFAULT_SCORES_FILENAME, DEFAULT_CAL_PARAMS_FILENAME, DEFAULT_PARAMS_FILENAME, DEFAULT_MODEL_FILENAME
from optswmm.utils.calparams import CalParams


XAXIS_CHOICES = ["iter", "datetime"]

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

import plotly.express as px



def _plot_param_changes(df, xaxis="iter"):
    # Plot all parameter values over time, colored by parameter name
    if xaxis not in XAXIS_CHOICES:
        raise ValueError(f"xaxis must be one of {XAXIS_CHOICES}")


    fig = px.line(
        df.reset_index(),
        x=xaxis,
        y="cal_val",
        color="param_tag",
        title="Parameter Physical Values Over Time",
        labels={"physical_val": "Physical Value", "datetime": "Datetime", "param_tag": "Parameter"}
    )
    fig.show()

def _plot_run_minimization(df: pd.DataFrame, xaxis="iter", smooth: float = 0):
    """
    Plot the minimization scores from a DataFrame or directory of runs.
    :param df: DataFrame containing run scores or path to a directory with run results.
    :type df: Path, str, or pd.DataFrame
    :param smooth: Smoothing factor between 0 (no smoothing) and 1 (maximum smoothing).
    :type smooth: float
    :raises ValueError: If the DataFrame does not contain 'score' and 'iter' columns or is empty.
    """
    if xaxis not in XAXIS_CHOICES:
        raise ValueError(f"xaxis must be one of {XAXIS_CHOICES}")

    if not all(col in df.columns for col in ["score", "iter"]):
        raise ValueError("DataFrame must contain 'score' and 'iter' columns.")

    if df.empty:
        raise ValueError("DataFrame is empty. No runs to plot.")

    def smooth_series(series, factor):
        """
        Smooth a pandas Series using a rolling mean.
        factor: float between 0 and 1, where 0 = no smoothing, 1 = max smoothing (window = len(series))
        """
        if not 0 <= factor <= 1:
            raise ValueError("factor must be between 0 and 1")
        window = max(1, int(len(series) * factor))
        return series.rolling(window=window, min_periods=1).mean()

    fig = go.Figure()
    if "node" in df.columns:
        for node in df["node"].unique():
            node_df = df[df["node"] == node].sort_values(xaxis)
            x_vals = node_df[xaxis].values
            y_vals = node_df["score"].values
            if smooth > 0:
                y_vals = smooth_series(pd.Series(y_vals), smooth).values
            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f"Node: {node}"
            )
            fig.add_trace(trace)
    else:
        df_sorted = df.sort_values(xaxis)
        x_vals = df_sorted[xaxis].values
        y_vals = df_sorted["score"].values
        if smooth > 0:
            y_vals = smooth_series(pd.Series(y_vals), smooth).values
        trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name="Run Scores"
        )
        fig.add_trace(trace)

    # Mean score trace (smoothed if requested)
    mean_scores = df.groupby(xaxis)["score"].mean().sort_index()
    mean_x = mean_scores.index.values
    mean_y = mean_scores.values
    if smooth > 0:
        mean_y = smooth_series(mean_scores, smooth).values
    trace = go.Scatter(
        x=mean_x,
        y=mean_y,
        mode='lines',
        name="Mean Score",
        line=dict(dash='dash', width=2, color='black')
    )
    fig.add_trace(trace)

    fig.update_layout(
        title="Run Scores",
        xaxis_title="Iteration" if xaxis == "iter" else "Datetime",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend_title="Node" if "node" in df.columns else "Run"
    )
    fig.show()
    


# Example usage:
# plot_run_scores(scores_df)
class OptRun:
    """
    Class to handle the optimization run, including initialization and plotting of results.
    """

    def __init__(self, run_loc: Path, load_results:bool=True):
        self.dir = run_loc
        self.name = run_loc.name
        self.config = self._get_config()

        try:
            self.date = datetime.strptime(self.name.split("_")[1], "%d-%m-%y-%H%M%S")
        except Exception:
            self.date = None
        
        # option to not load results, in case you just want a big run summary
        if load_results:
            self.get_scores()
            self.get_params()
        else:
            self.scores = None
            self.params = None

    def _get_config(self):
        """
        Load the configuration dictionary from the run directory.
        """
        config_file = self.dir / "config.yml"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
        
        return load_dict(config_file)
    

    def get_timeseries(self, ii=None):
        """
        Find all files matching the pattern 'obs_{ii}.csv' in the timeseries directory.
        If ii is None, return all matching files. Otherwise, return the file for the given ii.
        """
        timeseries_dir = self.dir / "timeseries"

        iters_available = list(timeseries_dir.glob("sim_*.pkl"))
        iters_available = [jj.stem.split("_")[1] for jj in iters_available]

        iters_available = np.sort(np.array(iters_available, dtype=int))

        if ii is None:
            ii = iters_available[-1]  # Get the last iteration by default

        if ii not in iters_available:
            # Find the nearest available iteration

            ii_nearest = iters_available[np.searchsorted(iters_available, ii) - 1]
            warnings.warn(f"Iteration {ii} not found, using nearest available iteration {ii_nearest}.")
            ii = ii_nearest

        obs = pd.read_pickle(timeseries_dir / f"obs_{ii}.pkl")
        sim = pd.read_pickle(timeseries_dir / f"sim_{ii}.pkl")
        return obs, sim
    

    def best_iter(self):
        """
        Get the best iteration based on the minimum score.
        """
        if self.scores is None or self.scores.empty:
            raise ValueError("No scores available to determine best iteration.")
        
        best_iter = self.scores.groupby("iter")["score"].mean().idxmax()
        return best_iter
    
    def get_scores(self):
        """
        Get the scores from the optimization run.
        """
        scores_file = self.dir / "results_scores.txt"
        if not scores_file.exists():
            raise FileNotFoundError(f"Scores file {scores_file} does not exist.")

        scores = pd.read_csv(scores_file, sep=",", header=0, index_col=0).rename(columns=lambda x: x.strip())
        self.scores = scores

    def get_params(self):
        """
        Get the parameters from the optimization run.
        """
        params_file = self.dir / DEFAULT_PARAMS_FILENAME
        if not params_file.exists():
            self.params = None
            return None
            
        params = pd.read_csv(params_file)
        if "datetime" in params.columns:
            params.set_index(pd.DatetimeIndex(params["datetime"]), inplace=True)
            params.drop(columns=["datetime"], inplace=True)

        # load the params metadata to match the param id (ii) to the parameter name
        cal_params_file = self.dir / "calibration_parameters.csv"
        if not cal_params_file.exists():
            self.params = params
            return params
            
        param_index = pd.read_csv(cal_params_file, index_col=0)
        param_index = param_index.set_index("ii")
        param_index.loc[param_index["element"].isnull(), "element"] = "lumped"
        param_index["param_tag"] = [f"{p['section']}_{p['attribute']}_{p['element']}" for _, p in param_index.iterrows()]


        for row, name in param_index[param_index.row_attribute.notna()].iterrows():
            param_index.at[name.name, "param_tag"] = f"{param_index.at[name.name, 'param_tag']}_{param_index.at[name.name, 'row_attribute']}"


        params["param_tag"] = params["ii"].map(param_index["param_tag"])
        params["element"] = params["ii"].map(param_index["element"])
        params["section"] = params["ii"].map(param_index["section"])
        params["attribute"] = params["ii"].map(param_index["attribute"])
        params["row_attribute"] = params["ii"].map(param_index["row_attribute"])
        params["col_index"] = params["ii"].map(param_index["col_index"])
        params["row_index"] = params["ii"].map(param_index["row_index"])
        self.params = params
        return params

    def load_simulation_preconfig(self, model:Path|Model, iter=None):
        """
        Load simulation preconfiguration for a specified optimization iteration.

        :param model: Path to the SWMM model file.
        :type model: Path
        """
        if iter is None:
            iter = self.best_iter()

        # Find the nearest available iteration in self.params["iter"]

        available_iters = np.sort(self.params["iter"].unique())
        nearest_idx = np.searchsorted(available_iters, iter)
        if nearest_idx == len(available_iters):
            nearest_idx -= 1
        nearest_iter = available_iters[nearest_idx]
        params = self.params[self.params["iter"] == nearest_iter].copy()

        cal_params_file = self.dir / "calibration_parameters.csv"
        cp = CalParams().from_df(cal_params_file)
        cp.set_values(params.cal_val.values)
        return cp.make_simulation_preconfig(model=model)


    def plot_scores(self, xaxis="iter", smooth=0):
        """
        Plot the scores of the optimization run.
        :param xaxis: The x-axis to use for the plot. Options are 'iter' or 'datetime'.
        :type xaxis: str
        """
    
        _plot_run_minimization(self.scores, xaxis=xaxis, smooth=smooth)

    def plot_param_changes(self, xaxis="iter"):
        """
        Plot the parameter minimization results.
        :param xaxis: The x-axis to use for the plot. Options are 'iter' or 'datetime'.
        :type xaxis: str
        """
        _plot_param_changes(self.params, xaxis=xaxis)

class OptRuns:
    """
    Iterable collection of OptRun objects.
    """

    def __init__(self, runs_dir: Path, load_results:bool=False, tag:str=None):
        if isinstance(runs_dir, str):
            runs_dir = Path(runs_dir)
        if not runs_dir.is_dir():
            raise ValueError("runs_dir must be a directory containing run folders.")
        
        run_dirs = []

        # only include directories that have a scores file, otherwise they failed on initialization
        for run in runs_dir.iterdir():
            if run.is_dir() and (run / DEFAULT_SCORES_FILENAME).exists():
                run_dirs.append(run)
        self.run_dirs = run_dirs

        if tag is not None:
            self.run_dirs = [run_dir for run_dir in self.run_dirs if tag in run_dir.name]
            
        self.load_results = load_results  # Default to loading results

        self.runs = [OptRun(run_loc=run_dir, load_results=load_results) for run_dir in self.run_dirs]

        # if loading results, filter results with empty score
        if self.load_results:
            self.runs = [r for r in self.runs if r.scores is not None and not r.scores.empty]
                
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
            "model_file": [run.config.get("model_file", None) for run in self.runs],
            "algorithm": [run.config.get("algorithm", None) for run in self.runs],
            "hierarchical": [run.config.get("hierarchical", False) for run in self.runs],
            "normalize": [run.config.get("normalize", None) for run in self.runs],
            "target_variables": [', '.join(run.config.get("target_variables", [])) for run in self.runs],
            "score_function": [', '.join(run.config.get("score_function", [])) for run in self.runs],
            "max_iter": [run.config.get("algorithm_options", {}).get("maxiter", None) for run in self.runs],
            #"scores": [run.scores for run in self.runs],
            #"params": [run.get_params() for run in self.runs]
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


    def plot_runs_minimization(self):
        """
        Plot the minimization scores from a DataFrame or directory of runs.

        :param df: DataFrame containing run scores or path to a directory with run results.
        :type df: Path, str, or pd.DataFrame
        :raises ValueError: If the DataFrame does not contain 'score' and 'iter' columns or is empty.
        """
        runs_df = self.get_df()
        if not all(col in runs_df.columns for col in ["score", "iter"]):
            raise ValueError("DataFrame must contain 'score' and 'iter' columns.")

        for run_dir in runs_df.index:
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

