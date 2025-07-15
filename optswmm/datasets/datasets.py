

import pandas as pd



class Dataset(pd.DataFrame):
    def __init__(self, data):
        super().__init__()
        self =pd.DataFrame.__init__(self, data)

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method")

    def process_data(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_data(self):
        return self.data

class CamusTo(Dataset):
    def load_data(self):
        # Implement the logic to load data
        self.data = "Loaded data"

    def process_data(self):
        # Implement the logic to process data
        self.data = self.data.upper()


"""
variable [precip, stage, etc]
unit [mm/h, m, etc]
station id [1, 2, 3, etc]

precip_mm




"""

param = "precipitation(mm)"

import re
def _parse_param(param):
    match = re.search(r'\((.*?)\)', param)
    if match:
        return param[:match.start()].strip(), match.group(1)
    return None, param.strip()

params = ["temperature(degC)","precipitation(mm)","discharge(m3/s)","stage(m)"]




PARAM_CHOICES = ["precipitation","temperature","discharge","stage"]


# Example usage
unit = _parse_param(param)
print(unit)  # Output: mm



params = ["precip","temp"]





PARAMETER_NAMES = ["precipitation","temperature","discharge","stage"]