"""Library of performance metrics for hydrological model evaluation."""

import numpy as np
import pandas as pd
import xarray as xr

def nse(obs,sim) -> float:
    """Nash-Sutcliffe Efficiency."""
    obs,sim = preprocess(obs,sim)
    return 1 - np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2)

def nse_beta(obs,sim) -> float:
    """Nash-Sutcliffe Efficiency Beta."""
    obs,sim = preprocess(obs,sim)
    return (sim.mean() - obs.mean()) / obs.std()

def nse_alpha(obs,sim) -> float:
    """Nash-Sutcliffe Efficiency Alpha."""
    obs,sim = preprocess(obs,sim)
    return sim.std() / obs.std()

def nse_hf(obs,sim,p=95) -> float:  
    """Nash-Sutcliffe Efficiency High Flow."""
    obs,sim = preprocess(obs,sim)
    hf = np.percentile(obs,p)
    ind_eval = obs >= hf
    sim = sim[ind_eval]
    obs = obs[ind_eval]
    return 1 - np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2)

def nse_lf(obs,sim,p=5) -> float:
    """Nash-Sutcliffe Efficiency Low Flow."""
    obs,sim = preprocess(obs,sim)
    lf = np.percentile(obs,p)
    ind_eval = obs <= lf
    sim = sim[ind_eval]
    obs = obs[ind_eval]
    return 1 - np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2)

def n(obs,sim) -> int:
    """Number of common samples."""
    obs,sim = preprocess(obs,sim)
    return len(obs)

def mve(obs,sim) -> float:
    """Mean volume Error."""
    obs,sim = preprocess(obs,sim)
    return np.mean(obs-sim)

def kge(obs,sim) -> float:
    """Kling-Gupta Efficiency."""
    obs,sim = preprocess(obs,sim)
    sim_mean = np.mean(sim, axis=0)
    obs_mean = np.mean(obs)
    r_num = np.sum((sim - sim_mean) * (obs - obs_mean), axis=0)
    r_den = np.sqrt(np.sum((sim - sim_mean) ** 2, axis=0) * np.sum((obs - obs_mean) ** 2))

    if np.isnan(r_den) | (r_den == 0):
        return np.nan
    r = r_num / r_den
    alpha = np.std(sim, axis=0) / np.std(obs)
    beta = (np.sum(sim, axis=0) / np.sum(obs))
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def mse(obs,sim) -> float:
    """Mean Squared Error."""
    obs,sim = preprocess(obs,sim)
    return np.mean((obs-sim)**2)

def nmse(obs,sim) -> float:
    """Normalized Mean Squared Error."""
    obs,sim = preprocess(obs,sim)
    return np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2)

def rmse(obs,sim) -> float:
    """Root Mean Squared Error."""
    return mse(obs,sim)**0.5

def nrmse(obs,sim) -> float:
    obs,sim = preprocess(obs,sim)
    return rmse(obs,sim) / np.mean(obs)

def mae(obs,sim) -> float:
    """Mean Absolute Error."""
    obs,sim = preprocess(obs,sim)
    return np.mean(np.abs(obs-sim))

def pi(obs,sim,leadtime=1) -> float:
    """Persistence Index"""
    # for PI, important to lag observations prior to removing nans; need regular timestep
    if isinstance(obs,np.ndarray) or isinstance(obs,xr.DataArray):
        fp = np.roll(obs,leadtime)
        fp[:leadtime] = np.nan
        
    elif isinstance(obs, pd.DataFrame) or isinstance(obs, pd.Series):
        fp = np.array(obs.shift(leadtime))
    else:
        raise ValueError
        
    obs = np.array(obs)
    sim = np.array(sim)
    
    nind = np.isnan(np.concatenate([obs.reshape(-1,1),sim.reshape(-1,1),fp.reshape(-1,1)],axis=1)).any(axis=1)
    fp = fp[~nind]
    obs = obs[~nind]
    sim = sim[~nind]
    return 1 - np.sum((obs-sim)**2)/np.sum((obs-fp)**2)


def e_u(obs,sim) -> float:
    """Mean Error."""
    obs,sim = preprocess(obs,sim)
    return np.mean(obs-sim)

def e_std(obs,sim) -> float:
    """Standard Deviation of Error."""
    obs,sim = preprocess(obs,sim)
    return np.std(obs-sim)

def pve(obs,sim) -> float:
    """Percent Volume Error."""
    obs,sim = preprocess(obs,sim)
    return (np.sum(obs) - np.sum(sim))/np.sum(obs)

def pep(obs,sim) -> float:
    """Percent Peak Error."""
    obs,sim = preprocess(obs,sim)
    return (np.max(obs) - np.max(sim))/np.max(obs)

def peak_obs(obs,sim) -> float:
    """Peak Observed."""
    obs,sim = preprocess(obs,sim)
    return np.max(obs)

def peak_sim(obs,sim) -> float:
    """Peak Simulated."""
    obs,sim = preprocess(obs,sim)
    return np.max(sim)

def mean_obs(obs,sim) -> float:
    """Mean Observed."""
    obs,sim = preprocess(obs,sim)
    return np.mean(obs)

def mean_sim(obs,sim) -> float:
    """Mean Simulated."""
    obs,sim = preprocess(obs,sim)
    return np.mean(sim)

def preprocess(obs,sim):
    """Remove nans and convert to numpy arrays."""
    # convert to array and remove nans
    obs = np.array(obs)
    sim = np.array(sim)
    
    # if len(sim) == 1:
    #     sim = np.reshape(sim,[-1,1])
    
    nind = np.isnan(np.concatenate([obs.reshape(-1,1),sim.reshape(-1,1)],axis=1)).any(axis=1)
    if np.all(nind):
        Warning("All values are nan")
        return np.nan, np.nan
    obs = obs[~nind]
    sim = sim[~nind]
    
    return obs, sim


def load_standard_metrics():
    """Return a list of standard metrics."""
    return ['nse','nse_beta','nse_alpha','nse_hf','nse_lf','mae','rmse','mse','kge','mve','pi','e_std','e_u','pve','pep','peak_obs','mean_obs','mean_sim','peak_sim','nrmse']

def check_inputs(x,y):
    """Check if inputs are valid for performance evaluation."""

    if type(x) != type(y):
        raise ValueError('x and y must be of same type (dataframe or array)')
    if type(x) == pd.DataFrame:
        if ~(y.columns == x.columns).all():
            raise ValueError('columns of x and y must be identical')
    elif x.shape[1] != y.shape[1]:
        raise ValueError('number of columns in x and y must match')
    if type(x) == pd.DataFrame:
        if ~(y.index == x.index).all():
                raise ValueError('indices of x and y must be identical')
    elif x.shape[0] != y.shape[0]:
        raise ValueError('number of rows in x and y must match')


def get_performance(obs, sim, list_of_metrics=''):
    """Return a dataframe of performance metrics."""
    
    check_inputs(obs, sim)

    if len(list_of_metrics) == 0:
        list_of_metrics = load_standard_metrics()
    
    performance = {}
    
    if type(obs) == pd.DataFrame:
        cols = list(obs.columns)
    else:
        cols = range(obs.shape[1])

    for metric in list_of_metrics:
        res = list()
        for col in cols:
            res.append(eval(metric)(obs[col], sim[col]))
        performance[metric] = res
    

    df = pd.DataFrame(index=cols, columns=list_of_metrics)
    for key in performance.keys():
        df.loc[:,key] = performance[key]
    return df

