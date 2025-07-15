import numpy as np
import pandas as pd
from copy import deepcopy
from pyswmm import Simulation, SimulationPreConfig, Subcatchments, Nodes, Links

PARAM_INDICES = {
    "subcatchments": {
        "RainGage": 1,
        "Outlet": 2,
        "Area": 3,
        "Width": 4,
        "PercImperv": 5,
        "Slope": 6,
        "CurbLength": 7,
        "SnowPack": 8
    },
    "subareas": {
        "N-Imperv": 1,
        "N-Perv": 2,
        "S-Imperv": 3,
        "S-Perv": 4,
        "PctZero": 5,
        "RouteTo": 6
    },
    "infiltration": {
        "Param1": 1,
        "Param2": 2,
        "Param3": 3,

        "Param4": 4,
        "Param5": 5,
        "CurveNum": 1,
    },
    "junctions": {
        "Elevation": 1,
        "MaxDepth": 2,
        "InitDepth": 3,
        "SurDepth": 4,
        "Aponded": 5
    }
}

PARAM_TRANSLATIONS = {"subcatchments":{"PercImperv":"percent_impervious", "Width":"width"}}


class CalParam():
    def __init__(self, 
                 section: str,
                 attribute: str,
                 key:list[str]=["index"],
                 lower: float = 0.0, 
                 upper: float = np.inf, 
                 lower_bound: float = -np.inf, 
                 upper_bound: float = np.inf, 
                 distributed: bool=False):

        self.section = section
        self.attribute = attribute
        self.key = key
        self.lower = lower
        self.upper = upper
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.distributed = distributed

        self.initial_value = None
        self.value = None
        self.index = None
        self.row_num = None
        self.obj_id = None
        self._standardize()

    def _standardize(self):
         self.index = PARAM_INDICES[self.section][self.attribute]
         self.row_num = None # placeholder

    def copy(self):
        return deepcopy(self)


class CalParams():
    def __init__(self):
        self.params = []
    
    def add_param(self, param: CalParam):
        """
        Add a calibration parameter to the list.

        :param param: Calibration parameter to add
        :type param: CalParam
        """
        self.params.append(param)

    def get_params(self):
        """
        Retrieve all calibration parameters.

        :return: List of calibration parameters
        :rtype: list[CalParam]
        """
        return self.params

    def clear_params(self):
        """
        Clear all calibration parameters.
        """
        self.params.clear()


    def _distribute(self, model_file):

        model_file = str(model_file)

        unique_sections = np.unique([param.section for param in self.params])
        

        with Simulation(model_file) as sim:
            subcatchment_ids = [s.subcatchmentid for s in Subcatchments(sim)]
            node_ids = [n.nodeid for n in Nodes(sim)]
            link_ids = [l.linkid for l in Links(sim)]


            dcp = []
            for id in subcatchment_ids:
                for param in self.params:
                    if param.section == "subcatchments":
                        cp = param.copy()
                        cp.obj_id = id
                        attribute_name = PARAM_TRANSLATIONS[cp.section][cp.attribute]
                        cp.initial_value = Subcatchments(sim)[id].__getattribute__(attribute_name)
                        dcp.append(cp)
            
        return dcp


#    def get_spc_args(self):
#        return {"section":self.section, "obj_id"}
#        return self.section, self.attribute, self.index, self.row_num, self.lower, self.upper, self.lower_bound, self.upper_bound
