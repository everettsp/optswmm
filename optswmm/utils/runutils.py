import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from optswmm.utils.functions import load_dict
import plotly.graph_objs as go
import numpy as np
import warnings
from swmmio import Model
import shutil
from pyswmm import Simulation, SimulationPreConfig
from optswmm.defs.filenames import DEFAULT_SCORES_FILENAME, DEFAULT_CAL_PARAMS_FILENAME, DEFAULT_PARAMS_FILENAME, DEFAULT_MODEL_FILENAME
from optswmm.utils.calparams import CalParams
from optswmm.utils.fileutils import initialize_run
from optswmm.utils.optconfig import OptConfig

XAXIS_CHOICES = ["iter", "datetime"]

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


def _get_run_minimization(
    df: pd.DataFrame,
    xaxis="iter",
    smooth: float = 0,
    mean: bool = True,
    median: bool = False,
    distributed: bool = True,
    quantile=None,
):
    """
    Compute the minimization series from a DataFrame.
    Returns a dict with keys: 'distributed', 'mean', 'median', 'quantile'
    """
    if xaxis not in XAXIS_CHOICES:
        raise ValueError(f"xaxis must be one of {XAXIS_CHOICES}")

    if not all(col in df.columns for col in ["score", "iter"]):
        raise ValueError("DataFrame must contain 'score' and 'iter' columns.")

    if df.empty:
        raise ValueError("DataFrame is empty. No runs to process.")

    def smooth_series(series, factor):
        if not 0 <= factor <= 1:
            raise ValueError("factor must be between 0 and 1")
        window = max(1, int(len(series) * factor))
        return series.rolling(window=window, min_periods=1).mean()

    result = {}

    # Distributed scores
    if distributed:
        if "node" in df.columns:
            result["distributed"] = []
            for node in df["node"].unique():
                node_df = df[df["node"] == node].sort_values(xaxis)
                x_vals = node_df[xaxis].values
                y_vals = node_df["score"].values
                if smooth > 0:
                    y_vals = smooth_series(pd.Series(y_vals), smooth).values
                result["distributed"].append({"name": f"Node: {node}", "x": x_vals, "y": y_vals})
        else:
            df_sorted = df.sort_values(xaxis)
            x_vals = df_sorted[xaxis].values
            y_vals = df_sorted["score"].values
            if smooth > 0:
                y_vals = smooth_series(pd.Series(y_vals), smooth).values
            result["distributed"] = [{"name": "Run Scores", "x": x_vals, "y": y_vals}]

    # Quantile range
    if quantile is not None:
        if quantile < 0 or quantile > 1:
            raise ValueError("quantile must be between 0 and 1")
        if quantile >= 0 and quantile <= 0.5:
            quantile = 1 - quantile

        p10_scores = df.groupby(xaxis)["score"].quantile(1-quantile).sort_index()
        p90_scores = df.groupby(xaxis)["score"].quantile(quantile).sort_index()
        x_vals = p10_scores.index.values

        if smooth > 0:
            p10_scores = smooth_series(p10_scores, smooth).values
            p90_scores = smooth_series(p90_scores, smooth).values

        result["quantile"] = {
            "x": x_vals,
            "p10": p10_scores,
            "p90": p90_scores,
            "quantile": quantile
        }

    # Mean score
    if mean:
        mean_scores = df.groupby(xaxis)["score"].mean().sort_index()
        x_vals = mean_scores.index.values
        if smooth > 0:
            mean_scores = smooth_series(mean_scores, smooth).values
        result["mean"] = {"x": x_vals, "y": mean_scores}

    # Median score
    if median:
        median_scores = df.groupby(xaxis)["score"].median().sort_index()
        x_vals = median_scores.index.values
        if smooth > 0:
            median_scores = smooth_series(median_scores, smooth).values
        result["median"] = {"x": x_vals, "y": median_scores}

    return result


def _plot_run_minimization(
    df: pd.DataFrame,
    xaxis="iter",
    smooth: float = 0,
    mean: bool = True,
    median: bool = False,
    distributed: bool = True,
    quantile=None,
    fig=None
):
    """
    Plot the minimization scores from a DataFrame or directory of runs.
    """
    data = _get_run_minimization(
        df,
        xaxis=xaxis,
        smooth=smooth,
        mean=mean,
        median=median,
        distributed=distributed,
        quantile=quantile,
    )

    if fig is None:
        fig = go.Figure()

    # Distributed traces
    if "distributed" in data:
        for trace in data["distributed"]:
            fig.add_trace(go.Scatter(
                x=trace["x"],
                y=trace["y"],
                mode='lines',
                name=trace["name"]
            ))

    # Quantile fill
    if "quantile" in data:
        q = data["quantile"]
        fig.add_trace(go.Scatter(
            x=np.concatenate([q["x"], q["x"][::-1]]),
            y=np.concatenate([q["p90"], q["p10"][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="10th-90th Percentile"
        ))

    # Mean trace
    if "mean" in data:
        m = data["mean"]
        fig.add_trace(go.Scatter(
            x=m["x"],
            y=m["y"],
            mode='lines',
            name="Mean Score",
            line=dict(dash='dash', width=2, color='black')
        ))

    # Median trace
    if "median" in data:
        m = data["median"]
        fig.add_trace(go.Scatter(
            x=m["x"],
            y=m["y"],
            mode='lines',
            name="Median Score",
            line=dict(dash='dot', width=2, color='red')
        ))

    fig.update_layout(
        title="Run Scores",
        xaxis_title="Iteration" if xaxis == "iter" else "Datetime",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        legend_title="Node" if ("distributed" in data and len(data["distributed"]) > 1 and "Node" in data["distributed"][0]["name"]) else "Run"
    )

    return fig


# Example usage:
# plot_run_scores(scores_df)
class OptRun:
    """
    Class to handle the optimization run, including initialization and plotting of results.
    """

    def __init__(self, run_loc: Path, load_results:bool=True):
        self.dir = run_loc
        self.name = run_loc.name
        self.experiment_name = self.name.split("_")[0]  # Extract experiment name from the run directory name

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
        Load the configuration dictionary from the run directory using OptConfig class method.
        """
        config_file = self.dir / "config.yml"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
        return OptConfig.from_file(config_file)

    def get_timeseries(self, ii=None):
        """
        Find all files matching the pattern 'obs_{ii}.csv' in the timeseries directory.
        If ii is None, return all matching files. Otherwise, return the file for the given ii.
        """
        timeseries_dir = self.dir / "timeseries"

        iters_available = list(timeseries_dir.glob("sim_*.pkl"))
        iters_available = [jj.stem.split("_")[1] for jj in iters_available]

        iters_available = np.sort(np.array(iters_available, dtype=int))

        if iters_available.size == 0:
            raise FileNotFoundError(f"No timeseries files found in {timeseries_dir}")

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
    

    def get_best_iter(self, available_iters):
        """
        Get the best iteration based on the minimum score.
        """
        if self.scores is None or self.scores.empty:
            raise ValueError("No scores available to determine best iteration.")
        scores = self.scores[self.scores["iter"].isin(available_iters)]

        best_iter = scores.groupby("iter")["score"].mean().idxmax()
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


    def _get_param_score_matrix(self):
        scores = self.scores.set_index("iter")["score"].to_dict()
        params = self.params.set_index("iter")
        params = params.join(pd.Series(scores, name="score"))

        # scatter plot with score on x-axis and param_tag on y-axis (with jitter for visibility)
        params["iter"] = params.index

        df = pd.DataFrame(index=params.index.unique())

        for ii, row in params.iterrows():
            df.loc[row.name, row.param_tag] = row.physical_val
            df.loc[row.name, "score"] = row.score

        # move 'score' to the left-most column
        if "score" in df.columns:
            df = df[["score"] + [c for c in df.columns if c != "score"]]
        return df

    def get_param_feature_model(self, regressor="random_forest"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        REGRESSOR_CHOICES = ["random_forest", "linear", "neural_network"]

        if regressor not in REGRESSOR_CHOICES:
            raise ValueError(f"regressor must be one of {REGRESSOR_CHOICES}")

        df = self._get_param_score_matrix().dropna()
        predictors = [c for c in df.columns if c != "score"]
        X = df[predictors].values
        y = df["score"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if regressor == "random_forest":
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        elif regressor == "linear":
            model = LinearRegression()
        elif regressor == "neural_network":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

        model.fit(X_train, y_train)
        return model, predictors, X_test, y_test


    def mc_param_search(self, regressor="random_forest", n_samples=10000, random_state=0):
        """
        Perform Monte Carlo simulation on parameter space to estimate model score predictions.

        :param regressor: Type of regressor to use ("random_forest", "linear", "neural_network").
        :type regressor: str
        :param n_samples: Number of Monte Carlo samples.
        :type n_samples: int
        :param random_state: Random seed for reproducibility.
        :type random_state: int
        :return: DataFrame with sampled parameters and predicted scores.
        :rtype: pd.DataFrame
        """

        # Get trained model and predictors
        model, predictors, _, _ = self.get_param_feature_model(regressor=regressor)

        # Load calibration parameter bounds
        cal_params_file = self.dir / "calibration_parameters.csv"
        if not cal_params_file.exists():
            raise FileNotFoundError(f"Calibration parameters file {cal_params_file} does not exist.")
        cal_params = pd.read_csv(cal_params_file, index_col=0)

        # Prepare bounds for each predictor
        bounds = []
        for param in predictors:
            row = cal_params[cal_params["param_tag"] == param]
            if row.empty:
                # fallback: use min/max from params
                vals = self.params[self.params["param_tag"] == param]["cal_val"]
                bounds.append((vals.min(), vals.max()))
            else:
                bounds.append((row["min"].values[0], row["max"].values[0]))

        # Monte Carlo sampling
        rng = np.random.default_rng(random_state)
        samples = np.array([rng.uniform(low, high, n_samples) for low, high in bounds]).T

        # Predict scores
        scores = model.predict(samples)

        # Build result DataFrame
        df_samples = pd.DataFrame(samples, columns=predictors)
        df_samples["predicted_score"] = scores

        return df_samples

    def run_monte_carlo_and_get_best_model(self, regressor="random_forest", n_samples=10000, random_state=0):
        """
        Run Monte Carlo parameter sensitivity, get the best (lowest predicted score) sample,
        and create a calibrated model using those parameters.

        :param regressor: Type of regressor to use ("random_forest", "linear", "neural_network").
        :type regressor: str
        :param n_samples: Number of Monte Carlo samples.
        :type n_samples: int
        :param random_state: Random seed for reproducibility.
        :type random_state: int
        :return: Tuple of (best_params: dict, best_score: float, model: Model)
        """
        # Run Monte Carlo simulation
        df_samples = self.mc_param_search(
            regressor=regressor,
            n_samples=n_samples,
            random_state=random_state
        )

        # Find the best (lowest) predicted score
        best_idx = df_samples["predicted_score"].idxmin()
        best_row = df_samples.loc[best_idx]
        best_params = best_row.drop("predicted_score").to_dict()
        best_score = best_row["predicted_score"]

        # Load calibration parameter file to get mapping to ii
        cal_params_file = self.dir / "calibration_parameters.csv"
        cal_params = pd.read_csv(cal_params_file, index_col=0)
        param_tags = cal_params["param_tag"].values
        param_order = [best_params[tag] for tag in param_tags if tag in best_params]

        # Set up CalParams and create simulation preconfig
        cp = CalParams().from_file(cal_params_file)
        cp.set_values(param_order)

        # Create a calibrated model (requires base model file from config)
        base_model_file = self.config.get("model_file")
        if base_model_file is None:
            raise ValueError("Base model file not specified in config.")
        base_model_file = Path(base_model_file)
        new_model_file = base_model_file.with_name(base_model_file.stem + "_mc_best.inp")

        # Generate the new model
        spc = cp.make_simulation_preconfig(model=Model(base_model_file))
        outputfile = str(base_model_file.with_suffix('.out').resolve())
        with Simulation(str(base_model_file), outputfile=outputfile, sim_preconfig=spc) as sim:
            sim.execute()
        modified_file = base_model_file.with_name(base_model_file.stem + "_mod" + base_model_file.suffix)
        shutil.copy(modified_file, new_model_file)
        model = Model(str(new_model_file))

        return best_params, best_score, model



    def get_param_feature_importance(self, regressor="random_forest"):
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np

        model, predictors, X_test, y_test = self.get_param_feature_model(regressor=regressor)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        if regressor == "random_forest":
            feat_imp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=True)
        elif regressor == "linear":
            feat_imp = pd.Series(np.abs(model.coef_), index=predictors).sort_values(ascending=True)
        elif regressor == "neural_network":
            coefs = model.coefs_[0]  # shape: (n_features, n_hidden)
            feat_imp = pd.Series(np.sum(np.abs(coefs), axis=1), index=predictors).sort_values(ascending=True)

        return {"feature_importances": feat_imp, "r2": r2, "mse": mse}

    def load_simulation_preconfig(self, model:Path|Model, iter=None):
        """
        Load simulation preconfiguration for a specified optimization iteration.

        :param model: Path to the SWMM model file.
        :type model: Path
        """


        # Find the nearest available iteration in self.params["iter"]


        # here, we need to find the best iteration for each parameter
        # since some calibration methods don't modify all parameters at each iteration, we need to find the nearest available iteration for each parameter

        unique_params = self.params["param_tag"].unique()
        params = pd.DataFrame()
        for param in unique_params:
            param_subset = self.params[self.params["param_tag"] == param]
            available_iters = np.sort(param_subset["iter"].unique())

            if iter is None:
                iter = self.get_best_iter(available_iters)

            nearest_idx = np.searchsorted(available_iters, iter)
            if nearest_idx == len(available_iters):
                nearest_idx -= 1

            nearest_iter = available_iters[nearest_idx]
            param_iter = param_subset[param_subset["iter"] == nearest_iter]
            params = pd.concat([params, param_iter], ignore_index=True)

        cal_params_file = self.dir / "calibration_parameters.csv"
        cp = CalParams().from_df(cal_params_file)
        cp.set_values(params.cal_val.values)
        return cp.make_simulation_preconfig(model=model)

    def get_calibrated_model(self, base_model_file:Path, new_model_file:Path, iter:int|None=None):
        """
        Get the calibrated model for a specified optimization iteration.

        :param model: Path to the SWMM model file.
        :type model: Path
        """

        if not Path(base_model_file).exists():
            raise FileNotFoundError(f"Base model file {base_model_file} does not exist.")
        
        if not Path(new_model_file).parent.exists():
            raise FileNotFoundError(f"Directory for new model file {new_model_file} does not exist.")
        
        if not Path(new_model_file).suffix == ".inp":
            raise ValueError(f"New model file {new_model_file} must have .inp extension.")
        
        spc = self.load_simulation_preconfig(model=Model(base_model_file), iter=iter)

        # Run simulation with updated parameters
        outputfile = str(Path(base_model_file).with_suffix('.out').resolve())

        with Simulation(str(base_model_file), outputfile=outputfile, sim_preconfig=spc) as sim:
            sim.execute()


        modified_file = Path(base_model_file).with_name(Path(base_model_file).stem + "_mod" + Path(base_model_file).suffix)
        shutil.copy(modified_file, new_model_file)
        return Model(str(new_model_file))    

    def plot_scores(
        self,
        xaxis="iter",
        smooth=0,
        mean=True,
        median=False,
        distributed=True,
        quantile=None,
        fig=None
    ):
        """
        Plot the scores of the optimization run.
        :param xaxis: The x-axis to use for the plot. Options are 'iter' or 'datetime'.
        :type xaxis: str
        :param smooth: Smoothing factor between 0 (no smoothing) and 1 (maximum smoothing).
        :type smooth: float
        :param mean: Whether to plot the mean score.
        :type mean: bool
        :param median: Whether to plot the median score.
        :type median: bool
        :param distributed: Whether to plot distributed scores (by node).
        :type distributed: bool
        :param quantile: Quantile range to plot (e.g., 0.1 for 10th-90th percentile).
        :type quantile: float or None
        :param fig: Optional plotly Figure to add traces to.
        :type fig: plotly.graph_objs.Figure or None
        """
    
        return _plot_run_minimization(
            self.scores,
            xaxis=xaxis,
            smooth=smooth,
            mean=mean,
            median=median,
            distributed=distributed,
            quantile=quantile,
            fig=fig
        )

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

        self.runs = [OptRun(run_loc=run_dir, load_results=load_results) for run_dir in self.run_dirs if (run_dir / "config.yml").exists()]

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

