import os
from pathlib import Path
import re


def _standardize_ext(ext:str):
    if not isinstance(ext, str):
        raise ValueError("ext must be a string")
    if not ext.startswith("."):
        ext = "." + ext
    return ext

def _check_ext(filename:str, ext:str):
    """
    check if a file is a GeoTIFF
    :param filename: path to the file
    :type filename: str
    """
    ext = _standardize_ext(ext)
    
    if not Path(filename).suffix == ext:
        raise ValueError(f"DEM file {filename} must be a GeoTIFF.")

def _standardize_file(filename:str|Path, exists:bool=True, ext:str=""):
    """
    check if a file exists
    :param filename: path to the file
    :type filename: str
    """
    
    if isinstance(filename, str):
        filename = Path(filename)

    "if the file is in a zip archive, extract the zip filename for the exist() check"
    "an archived raster will typically have the form ''zip://path/to/file.zip!path/to/file.tif''"
    "this might throw an error if the zip archive uses a different format"
    if ".zip!" in str(filename):
        check_filename = re.sub(r"zip:\\(.*?\.zip)!.*", r"\1", str(filename))
    else:
        check_filename = filename

    if exists:
        if not Path(check_filename).exists():
            raise FileNotFoundError(f"File {check_filename} does not exist.")
        
    if ext != "":
        _check_ext(filename, ext)

    return filename
    
def _standardize_dir(directory:str|Path, exists:bool=True):
    """
    check if a directory exists
    :param directory: path to the directory
    :type directory: str
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if exists:
        if not Path(directory).exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
    return directory

import pandas as pd

def _check_same_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Check if two DataFrames have the same columns.

    :param df1: The first DataFrame.
    :type df1: pd.DataFrame
    :param df2: The second DataFrame.
    :type df2: pd.DataFrame
    :raises ValueError: If the columns of the two DataFrames are not the same.
    """
    if set(df1.columns.tolist()) != set(df2.columns.tolist()):
        raise ValueError('timeseries results must have the same columns (comparison points)')
    else:
        return
    
def sync_timeseries(ts1: pd.DataFrame, ts2: pd.DataFrame) -> pd.DataFrame:
    """
    Return the mutual timesteps of two pandas DataFrames with datetime indices.

    :param ts1: The first timeseries DataFrame.
    :type ts1: pd.DataFrame
    :param ts2: The second timeseries DataFrame.
    :type ts2: pd.DataFrame
    :return: The synchronized timeseries DataFrame.
    :rtype: pd.DataFrame
    :raises ValueError: If no mutual datetimes are found between the two timeseries.
    """
    _check_same_columns(ts1, ts2)
    mutual_datetimes = pd.merge(right=ts1, left=ts2, right_index=True, left_index=True).index

    if len(mutual_datetimes) == 0:
        raise ValueError('No mutual datetimes found between the two timeseries.')

    ts1 = ts1.loc[mutual_datetimes]
    ts2 = ts2.loc[mutual_datetimes]

    return ts1, ts2



def _standardize_pkl_data(data:pd.DataFrame | str | Path) -> pd.DataFrame:
    """
    validate that the target data is compatible with the model
    """


    if isinstance(data, pd.DataFrame):
        pass
    
    if isinstance(data, (str, Path)):
        data = _standardize_file(data, exists=True, ext="pkl")

        if not data.exists():
            raise ValueError(f"File {data} does not exist")

        data = pd.read_pickle(data)
        
    data.index = data.index.tz_localize(None)

    return data



from swmmio import Model
from utils.swmmutils import get_model_datetimes
import warnings

def _validate_target_data(tgt:pd.DataFrame | str | Path, model:Model) -> bool:
    """
    validate that the target data is compatible with the model
    param tgt: target data
    type tgt: pd.DataFrame
    param model: swmmio model object
    type model: Model - swmmio model object
    """
    #tgt = _standardize_pkl_data(data=tgt)
    
    if isinstance(tgt, str):
        tgt = Path(tgt)

    if isinstance(tgt, Path):
        if tgt.suffix == ".csv":
            tgt = pd.read_csv(tgt, index_col=0, parse_dates=True)
        elif tgt.suffix == ".pkl":
            tgt = pd.read_pickle(tgt)
        else:
            raise ValueError(f"Unsupported file extension: {tgt.suffix}")

    # check that all target stations are in the model
    model_nodes = model.nodes().index.tolist()
    
    if isinstance(tgt.columns, pd.MultiIndex):
        tgt_stations = tgt.columns.get_level_values('nodes').unique().to_list()

        missing_stations = [station for station in tgt_stations if station not in model_nodes]
        if missing_stations:
            raise ValueError(f"Missing stations in model: {missing_stations}")

    if len(tgt) < 100:
        warnings.warn(f"Target data '{tgt.columns}' has less than 100 timesteps", UserWarning)

    # check that the target data date range overlaps with the model date range
    #tgt_start_date, tgt_end_date = tgt.index.min(), tgt.index.max()
    #mdl_start_date, mdl_end_date = get_model_datetimes(model)
    
    # NOTE: temp fix, removing timezone specification for comparison here
    #tgt_start_date = tgt_start_date.replace(tzinfo=None)
    #tgt_end_date = tgt_end_date.replace(tzinfo=None)

    #if tgt_end_date < mdl_start_date or tgt_start_date > mdl_end_date:
    #    raise ValueError(f"No overlap between target data date range ({tgt_start_date} to {tgt_end_date}) and model date range ({mdl_start_date} to {mdl_end_date})")
    
    return True



