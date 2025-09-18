import json
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import pandas as pd

def invert_dict(d: dict) -> dict:
    """
    Invert the keys and values of a dictionary.

    :param d: The dictionary to invert.
    :type d: dict
    :return: The inverted dictionary.
    :rtype: dict
    """
    return {d[key]: key for key in d}

def save_dict(filename: Path, d: dict):
    """
    Save a dictionary to a file in JSON or YAML format.

    :param filename: The path to the file where the dictionary will be saved.
    :type filename: Path
    :param d: The dictionary to save.
    :type d: dict
    :raises NotImplementedError: If the file extension is not .yaml or .json.
    """
    if type(filename) == str:
        filename = Path(filename)
        
    if filename.suffix in ['.yml','.yaml']:
        with open(filename, 'w') as file:
            yaml.dump(d, file)
    elif filename.suffix == '.json':
        with open(filename, 'w+') as fp:
            json.dump(d, fp)
    else:
        raise NotImplementedError('only .yml, .yaml, and .json files are supported')

def load_dict(filename: Path) -> dict:
    """
    Load a dictionary from a JSON or YAML file.

    :param filename: The path to the file from which the dictionary will be loaded.
    :type filename: Path
    :return: The loaded dictionary.
    :rtype: dict
    :raises NotImplementedError: If the file extension is not .yaml or .json.
    """
    if type(filename) == str:
        filename = Path(filename)

    if filename.suffix in ['.yml','.yaml']:
        with open(filename, 'r') as file:
            d = yaml.safe_load(file)
    elif filename.suffix == '.json':
        
        with open(filename, 'r') as fp:
            d = json.load(fp)
    else:
        raise NotImplementedError('only .yml, .yaml and .json files are supported')
    return d

def check_same_columns(df1: pd.DataFrame, df2: pd.DataFrame):
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

    
    check_same_columns(ts1, ts2)
    ts2 = ts2[ts1.columns]

    # round to nearest minute, since sometimes SWMM results are off by 1 second
    # TODO: figure out why SWMM results are sometimes off by 1 second. The options all seem to be fine.
    
    ts1.index = ts1.index.round('T')
    ts2.index = ts2.index.round('T')
    mutual_datetimes = pd.merge(right=ts1, left=ts2, right_index=True, left_index=True).index

    if len(mutual_datetimes) == 0:
        raise ValueError('No mutual datetimes found between the two timeseries.')

    ts1 = ts1.loc[mutual_datetimes]
    ts2 = ts2.loc[mutual_datetimes]

    return ts1, ts2

def datetime_to_elapsed(datetime_index, multiplier=1):
    """
    Convert a datetime index to elapsed time.

    :param datetime_index: The datetime index to convert.
    :type datetime_index: pd.DatetimeIndex
    :param multiplier: A multiplier to apply to the elapsed time, defaults to 1.
    :type multiplier: int, optional
    :return: A list of elapsed time values.
    :rtype: list
    """
    elapsed = [(dt.minute + dt.hour*60 + dt.dayofyear * 24 * 60) + dt.year * 24 * 60 * 365 for dt in datetime_index]
    elapsed = [x * multiplier for x in elapsed]
    return [dt - elapsed[0] for dt in elapsed]



import re
def clean_gml_file(file_path):
    """
    Function to clean the NP.FLOAT64() wrappers from a GML file.

    Args:
    - file_path (str): Path to the GML file to be cleaned.

    Returns:
    - None: The file is modified in place.
    """
    try:
        # Read the content of the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Regular expression to match NP.FLOAT64() and extract the value
        content_cleaned = re.sub(r'NP\.FLOAT64\((.*?)\)', r'\1', content)

        # Write the cleaned content back to the same file
        with open(file_path, 'w') as file:
            file.write(content_cleaned)

        print(f"NP.FLOAT64() removed and values cleaned successfully in {file_path}!")

    except Exception as e:
        print(f"An error occurred: {e}")


