"""Additional SWMM functions, mainly for quickly fetching simulation results and running the model"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from datetime import datetime
from tqdm import tqdm
import swmmio
import yaml
from swmm.toolkit.shared_enum import SubcatchAttribute, NodeAttribute, LinkAttribute
from pyswmm import Output
from swmmio import Model




def aggregate_conduits_at_nodes(mdl:swmmio.Model, nodes:list[str]=None, aggregation:str="sum", attribute:str="FLOW_RATE") -> pd.DataFrame:
    """
    Retrieve aggrecated conduit results at each node. This aggregates the timeseries for upstream conduits at each node, or the downstream conduit at zero-order nodes.

    :param mdl: swmmio.Model
    :type mdl: swmmio.Model
    :param nodes: list of nodes for which to fetch results
    :type nodes: list[str]
    :param aggregation: aggregation function to use (default is "sum")
    :type aggregation: str
    :param attribute: attribute to aggregate (default is "FLOW_RATE")
    :type attribute: str
    :return: aggregated conduit results at each node
    :rtype: pd.DataFrame
        
    """

    #TODO: This could be sped up by only fetching the conduits for the selected nodes.
    df = get_link_timeseries(mdl)
    AGGR_FUNCTIONS = ["sum","median","mean"]
    aggregation = aggregation.lower()
    if aggregation not in AGGR_FUNCTIONS:
        raise ValueError(f"aggregation function '{aggregation}' not available, chocies include {AGGR_FUNCTIONS}")
    
    if attribute not in [attr.name for attr in LinkAttribute]:
        raise ValueError(f"param '{attribute}' not available, chocies include {[a.name for a in LinkAttribute]}")

    if not nodes:
         Warning("no node IDs provided, fetching all nodes..")
         nodes = mdl.nodes().index.tolist()

    dfs = {}
    for n in nodes:
        # by default, this grabs the upstream conduits at each node. If the node is located at a network extremity, it will take from the downstream conduit
        incoming_conduits = mdl.inp.conduits[n == mdl.inp.conduits.InletNode].index.tolist()
        if len(incoming_conduits) == 0:
            incoming_conduits = mdl.inp.conduits[n == mdl.inp.conduits.OutletNode].index.tolist()
        res = pd.DataFrame(getattr(df.loc[:,[("FLOW_RATE",c) for c in incoming_conduits]], aggregation)(axis=1), columns=[n])
        dfs[n] = res.copy()
        
    return pd.concat(list(dfs.values()), axis=1)


def get_link_timeseries(mdl:swmmio.Model, links:list[str]|str=None, multi_index:bool=True)->pd.DataFrame:
    """
    Return the simulation timeseries at links of a SWMM model.

    :param mdl: swmmio.Model
    :type mdl: swmmio.Model
    :param links: link ids for which to fetch timeseries
    :type links: list[str] | str
        
    :param multi_index: whether to return multi-index (node,attribute); if False will concatenate multiindex as 'node-attribute'
    :type multi_index: bool    
    :return: dataframe of simulation results at links
    :rtype: pd.DataFrame
    """

    dfs = []

    if links is None:
         Warning("no link IDs provided, fetching all links..")
         links = mdl.links().index.tolist()

    if type(links) == str:
         links = [links]

    out_file = get_model_path(mdl, ext='out')
    with Output(out_file) as out:
        for l in tqdm(links, leave=False):
            if l not in mdl.links().index:
                raise ValueError("link id not found")
            
            for param in [attr.name for attr in LinkAttribute]:
                    res = out.link_series(l, getattr(LinkAttribute,param))
                    dfs.append(pd.DataFrame(index=res.keys(), data=res.values(), columns=[(param,l)]))

    df = pd.concat(dfs, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    if not multi_index:
        df.columns = df.columns.to_series().apply(lambda x: "-".join(x))
    return df

def get_node_timeseries(
        model:swmmio.Model,
        nodes:list[str]|str=None,
        multi_index:bool=True,
        params:list[str]=None,
        show_progress:bool=False)->pd.DataFrame:
    """
    Return the simulation timeseries at nodes of a SWMM model.

    :param model: swmmio.Model
    :type model: swmmio.Model
    :param nodes: node ids for which to fetch timeseries
    :type nodes: list[str] | str
    :param multi_index: whether to return multi-index (node,attribute); if False will concatenate multiindex as 'node-attribute' 
    :type multi_index: bool
    :return: dataframe of simulation results at nodes
    :rtype: pd.DataFrame
    """

    dfs = []
    if type(nodes) is str:
         nodes = [nodes]

    if not params:
        Warning("no parameters provided, fetching all parameters..")
        params = [attr.name for attr in NodeAttribute]
    elif not isinstance(params, list):
        raise ValueError("params must be a list of strings")
    else:
        for param in params:
            if param not in [attr.name for attr in NodeAttribute]:
                raise ValueError(f"param '{param}' not available, chocies include {[attr.name for attr in NodeAttribute]}")

    if not nodes:
         Warning("no node IDs provided, fetching all nodes..")
         nodes = model.nodes().index.tolist()
    elif not isinstance(nodes, list):
        raise ValueError("nodes must be a list of strings")

    out_file = get_model_path(model, ext='out')
    with Output(out_file) as out:
        for n in tqdm(nodes, leave=False, disable=not show_progress):
            if n not in model.nodes().index:
                raise ValueError("node id not found")
            for param in params:
                    res = out.node_series(n, getattr(NodeAttribute,param))
                    dfs.append(pd.DataFrame(index=res.keys(), data=res.values(), columns=[(param,n)]))

    df = pd.concat(dfs, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    if not multi_index:
        df.columns = df.columns.to_series().apply(lambda x: "-".join(x))
    return df


def get_subcatchment_timeseries(
        mdl:swmmio.Model,
        subcatchments:list[str]|str=None,
        multi_index:bool=True,
        show_progress:bool=False)->pd.DataFrame:
    """
    Return the simulation timeseries at nodes of a SWMM model.

    :param mdl: swmmio.Model
    :type mdl: swmmio.Model
    :param subcatchments: subcatchment ids for which to fetch timeseries
    :type subcatchments: list[str] | str
    :param multi_index:  whether to return multi-index (node,attribute); if False will concatenate multiindex as 'node-attribute'
    :type multi_index: bool
    :return: dataframe of simulation results for subcatchments
    :rtpe: pd.DataFrame
    """

    dfs = []
    if type(subcatchments) is str:
         subcatchments = [subcatchments]

    if not subcatchments:
         Warning("no node IDs provided, fetching all nodes..")
         subcatchments = mdl.subcatchments().index.tolist()

    out_file = get_model_path(mdl, ext='out')
    with Output(out_file) as out:
        for n in tqdm(subcatchments, leave=False, disable=not show_progress):
            if n not in mdl.subcatchments().index:
                raise ValueError("node id not found")
            for param in [attr.name for attr in SubcatchAttribute]:
                    res = out.subcatch_series(n, getattr(SubcatchAttribute,param))
                    dfs.append(pd.DataFrame(index=res.keys(), data=res.values(), columns=[(param,n)]))

    df = pd.concat(dfs, axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    if not multi_index:
        df.columns = df.columns.to_series().apply(lambda x: "-".join(x))
    return df




def assign_default_parameter_values(df, section):
    if section.upper() in ['CURVE_NUMBER','GREEN_AMPT','HORTON','MODIFIED_GREEN_AMPT','MODIFIED_HORTON']:
        vals = DEFAULT_VALS['infiltration_cols'][section.upper()]

    else:
        vals = DEFAULT_VALS['inp_file_objects'][section.upper()]
    # convert list of dicts to dict
    # TODO: just fix the yaml file
    default_values = dict(pair for d in vals for pair in d.items())

    for col in df.columns:
        if df[col].isnull().all():
            if col in list(default_values.keys()):
                df[col] = default_values[col]
    return df

def dataframe_to_dat(filename, df):
    with open(filename, "w") as f:
        f.write("everett generated rainfall dat file, format: Station|Year|Month|Day|Hour|Minute|Precipitation\n")
        for col in df.columns:
            for dt in df.index:
                line = '{id:s}\t{year:04}\t{month:02d}\t{day:02d}\t{hour:02d}\t{minute:02d}\t{value:08f}'.format(id=col, year=dt.year, month=dt.month ,day=dt.day, hour=dt.hour, minute=dt.minute, value=df.loc[dt,col])
                f.write(line+"\n")


from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def run_swmm(inp_path:Path, verbose:bool=False):
    """
    Execute a SWMM model using the command line interface. This function will execute the SWMM model and save the results to the same directory as the input file.

    :param inp_path: path to the SWMM input file
    :type inp_path: Path
    :param verbose: whether to print the command output (default is False)
    :type verbose: bool
    """
    if type(inp_path) == str:
        inp_path = Path(inp_path)
        
    # if a swmmio.Model is used as input, grab the complete path to the inp file
    if type(inp_path) == swmmio.core.Model:
        inp_path = (Path(inp_path.inp.dir) / "{}.inp".format(inp_path.inp.name)).resolve()

    with open('./defs/config.yaml', 'r') as file:
        x = yaml.safe_load(file)
    
    swmm_exe = x["path_exec_swmm"]

    rpt_path = os.path.splitext(inp_path)[0] + '.rpt'
    out_path = os.path.splitext(inp_path)[0] + '.out'
    
    #command = '"{}" "{}" "{}" "{}"'.format(swmm_exe, inp_path, rpt_path, out_path)
    # DC: TODO : ADD QUIET MODE if possible
    # https://rdrr.io/github/scheidan/SWMMR/man/runSWMM.html
    
    command = list()
    command.insert(0, swmm_exe)
    command.insert(1, inp_path)
    command.insert(2, rpt_path)
    command.insert(3, out_path)
    #print("command = ", command,"\n\n") # DEBUG

    if verbose:
        subprocess.run(command)
    else:
        subprocess.run(command,capture_output=True, text=True).stdout.strip("\n")


def foo_runswmm(inp_path:Path):
    
    # DEBUG FOR WINDOWS STEP BY STEP...
    
    # if a swmmio.Model is used as input, grab the complete path to the inp file
    if type(inp_path) == swmmio.core.Model:
        inp_path = (Path(inp_path.inp.dir) / "{}.inp".format(inp_path.inp.name)).resolve()

    swmm_exe = "C:/Program Files/EPA SWMM 5.2.4 (64-bit)/runswmm.exe" # x["path_exec_swmm"]

    rpt_path = os.path.splitext(inp_path)[0] + '.rpt'
    out_path = os.path.splitext(inp_path)[0] + '.out'
    
    command = list()
    command.insert(0, swmm_exe)
    command.insert(1, inp_path)
    command.insert(2, rpt_path)
    command.insert(3, out_path)
    #print("command = ", command,"\n\n") # DEBUG
    subprocess.run(command)    

from defs import SWMM_DATETIME_FMT


def get_model_datetimes(model:swmmio.Model):
    start_datetime = model.inp.options.loc["START_DATE"].Value + " " + model.inp.options.loc["START_TIME"].Value
    start_datetime = datetime.strptime(start_datetime, SWMM_DATETIME_FMT)
    end_datetime = model.inp.options.loc["END_DATE"].Value + " " + model.inp.options.loc["END_TIME"].Value
    end_datetime = datetime.strptime(end_datetime, SWMM_DATETIME_FMT)
    return start_datetime, end_datetime



def set_model_datetimes(model: Model, start_datetime=None, end_datetime=None, report_step=None) -> Model:
    """
    Set the simulation start and end times in the model.

    :param model: Model to be updated.
    :type model: swmmio.Model object
    :param start_time: Simulation start time.
    :type start_time: datetime object
    :param end_time: Simulation end time.
    :type end_time: datetime object
    :returns: Model with updated simulation times.
    :rtype: swmmio.Model object
    """
    if start_datetime:
        model.inp.options.loc['START_TIME'] = datetime.strftime(start_datetime, format='%H:%M:%S')
        model.inp.options.loc['START_DATE'] = datetime.strftime(start_datetime, format='%m/%d/%Y')
        model.inp.options.loc['REPORT_START_TIME'] = datetime.strftime(start_datetime, format='%H:%M:%S')
        model.inp.options.loc['REPORT_START_DATE'] = datetime.strftime(start_datetime, format='%m/%d/%Y')

    if end_datetime:
        model.inp.options.loc['END_TIME'] = datetime.strftime(end_datetime, format='%H:%M:%S')
        model.inp.options.loc['END_DATE'] = datetime.strftime(end_datetime, format='%m/%d/%Y')

    if report_step:
        model.inp.options.loc['REPORT_STEP'] = datetime.strftime(report_step, '%H:%M:%S')

    return model


def get_model_path(model:swmmio.Model, ext:str='inp', as_str=True) -> str:
    """returns the complete (absolute) filepath of a swmmio model object"""
    if isinstance(model, (Path, str)):
        print('model is already a path')
        return model
    valid_extensions = ['inp','rpt','out']
    if ext not in valid_extensions:
        raise ValueError("'ext' parameter must be one of {}".format(" ".join(valid_extensions)))
    
    model_path = (Path(model.inp.dir) / "{}.{}".format(model.inp.name, ext)).resolve()
    if as_str:
        model_path = str(model_path)
    return model_path


def get_predictions_at_nodes(model:swmmio.Model,nodes:list[str]=None,param:str='FLOW_RATE') -> pd.DataFrame:
    """
    Aggregates the flows in the links upstream of each node.

    :param model: swmmio.Model
    :type model: swmmio.Model
    :param nodes: list of str: list of nodes for which to fetch predictions
    :type nodes: list[str]
    :param param: str: parameter to aggregate (default is 'FLOW_RATE')
    :type param: aggregated flows at each node
    :return: predictions at each node
    :rtype: pd.DataFrame
    """

    if param not in [attr.name for attr in LinkAttribute]:
        raise ValueError(f"param '{param}' not available, chocies include {[attr.name for attr in LinkAttribute]}")

    """aggregates the flows in the links upstream of each node"""
    if nodes is None:
        nodes = model.nodes.dataframe.index
    swmm_results = {}
    out_file = get_model_path(model, ext='out')
    with Output(out_file) as out:
        for n in nodes:
            incoming_conduits = model.inp.conduits[n == model.inp.conduits.InletNode].index.tolist()
            if len(incoming_conduits) == 0:
                incoming_conduits = model.inp.conduits[n == model.inp.conduits.OutletNode].index.tolist()

            dfs = []
            for c in incoming_conduits:
                res = out.link_series(c, getattr(LinkAttribute,param))
                dfs.append(pd.DataFrame(index=res.keys(), data=res.values(), columns=[c]))
            df = pd.concat(dfs, axis=1).sum(axis=1)
            swmm_results[n] = df
                    
    swmm_flows = pd.concat(swmm_results, axis=1)
    if swmm_flows.index[0].year < 1000:
        swmm_flows['corrected_datetime'] = [datetime(day=dt.day, month=dt.month, year=dt.year + 2000, hour=dt.hour, minute=dt.minute, second=dt.second) for dt in swmm_flows.index]
        swmm_flows = swmm_flows.set_index('corrected_datetime')
    swmm_flows.index.name = 'datetime'
    return swmm_flows



def flow_depth_to_AP(depth: np.array, height: np.array, width:np.array=None, shapes:str='DEFAULT') -> np.array:
    """
    Calculate the cross-sectional area and wetted perimeter of a flow given the depth, height, and width of the conduit. This function is vectorized to handle multiple shapes at once.
    
    :param depth: np.array
    :type depth: np.array
    :param height: np.array
    :type height: np.array
    :param width: np.array
    :type width: np.array
    :param shapes: shape of conduit, choices include ['circ','rect']
    :type shapes: str
    :return: cross-sectional area of flow and wetted perimeter
    :rtype: np.array
        
    """
    A_ret = np.zeros_like(height).astype(float)
    P_ret = np.zeros_like(height).astype(float)

    # applies a nan mask to vectorize calcs for each unique shape
    for shape in np.unique(shapes):
        idx = shape == shapes
        #if np.any(depth[idx] > height[idx]): raise ValueError('depth of flow cannot exceed conduit height')
        
        h = height[idx] - depth[idx]
        if shape.lower() in ['circ','circular','round']:
            r = height[idx]/2
            theta = 2*np.arccos((r-h)/r)
            A_pipe = np.pi * r**2
            A = np.pi * r**2 - ((r**2) * (theta - np.sin(theta)))/2
            P_pipe = 2 * np.pi * r
            P = P_pipe - r * theta

        elif shape.lower() in ['rect','rectangle']:
            if np.any(width[idx] == None): raise ValueError('{} type cross-section needs width parameter'.format(shape))
            A = depth[idx] * width[idx]
            P = depth[idx] * 2 + width[idx]
        else:
            raise NotImplementedError(f"{shape} not implemented")
        
        A_ret[idx] = A
        P_ret[idx] = P
        return A_ret, P_ret