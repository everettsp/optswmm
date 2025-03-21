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
        return data
    
    if isinstance(data, (str, Path)):
        data = _standardize_file(data, exists=True, ext="pkl")

    if not data.exists():
        raise ValueError(f"File {data} does not exist")
    
    data = pd.read_pickle(data)

    return data
