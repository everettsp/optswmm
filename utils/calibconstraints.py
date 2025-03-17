"""TEST"""

import pandas as pd
import numpy as np
from copy import deepcopy
from defs import SWMM_SECTION_SUBTYPES
from utils.networkutils import get_upstream_nodes
import swmmio

#DEFAULT_CHANGES = ['relative', 'absolute']

def tuple_to_str(tup: tuple) -> str:
    """
    Converts a tuple to a string.

    :param tup: Tuple to convert
    :type tup: tuple
    :return: String
    :rtype: str
    """
    return '_'.join([str(x) for x in tup])



class CalParam():
    def __init__(self, 
                 section: str, 
                 attribute: str, key:list[str]=["index"], 
                 lower: float = 0.0, 
                 upper: float = np.inf, 
                 lower_bound: float = -np.inf, 
                 upper_bound: float = np.inf, 
                 distributed: bool=False):
        
        """
        Initialize a parameter for SWMM calibration.

        :param section: Section of SWMM INP file (e.g., subcatchments)
        :type section: str
        :param attribute: Attribute of the section (e.g., slope)
        :type attribute: str
        :param lower: Lower constraint of search-space
        :type lower: float
        :param upper: Upper constraint of search-space
        :type upper: float
        :param lower_bound: Physics-based lower constraint for value (e.g., inflow >= 0)
        :type lower_bound: float
        :param upper_bound: Physics-based upper constraint for value
        :type upper_bound: float
        :raises NotImplementedError: If section is not found in SWMM_SECTION_SUBTYPES
        """

        if type(key) is str:
            key = [key]
            
        self.tag = ''
        self.section = section
        self.attribute = attribute
        self.element = None
        self.key = key
        self.parent_section = ''
        self.lower = lower
        self.upper = upper
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.distributed = distributed
        #self.distributed = False
        self._standardize()

    def copy(self):
        """
        Returns a deep copy of Constraint.

        :return: Deep copy of Constraint
        :rtype: Constraint
        """
        return deepcopy(self)

    def _standardize(self):
        """
        Standardizes the constraints.

        :raises ValueError: If change is not recognized or distributed is not a boolean
        """
        #if self.change not in DEFAULT_CHANGES:
        #    raise ValueError(f"change '{self.change}' not recognised. Choices: {DEFAULT_CHANGES}")
        #
        if self.section not in list(SWMM_SECTION_SUBTYPES.keys()):
            raise NotImplementedError(f"section '{self.section}' not found in SWMM_SECTION_SUPTYPES. If section is spelled correctly, might need to update DEFS. Currently implemented mappings: {list(SWMM_SECTION_SUBTYPES.keys())}.")
        
        self.parent_section = SWMM_SECTION_SUBTYPES[self.section]

        #if not isinstance(self.distributed, bool):
        #    raise ValueError(f"constraints 'distributed' parameter must be of type bool")

    def make_multi_index(self, model) -> pd.MultiIndex:
        index = [getattr(getattr(model.inp, self.section),x).to_list() for x in self.key]
        index = pd.MultiIndex.from_arrays(index)
        return index

    def append_parameter_constraint(self, section: str, attribute: str, key:list[str]=["index"], lower: float = 0.0, upper: float = np.inf, lower_bound: float = -np.inf, upper_bound: float = np.inf, distributed: bool=False):
        """
        Initialize a parameter for SWMM calibration.

        :param section: Section of SWMM INP file (e.g., subcatchments)
        :type section: str
        :param attribute: Attribute of the section (e.g., slope)
        :type attribute: str
        :param lower: Lower constraint of search-space
        :type lower: float
        :param upper: Upper constraint of search-space
        :type upper: float
        :param lower_bound: Physics-based lower constraint for value (e.g., inflow >= 0)
        :type lower_bound: float
        :param upper_bound: Physics-based upper constraint for value
        :type upper_bound: float
        :raises NotImplementedError: If section is not found in SWMM_SECTION_SUBTYPES
        """

        if type(key) is str:
            key = [key]

        tag = "{}.{}".format(section, attribute)
        self.df.loc[tag, 'section'] = section
        self.df.loc[tag, 'attribute'] = attribute
        self.df.loc[tag, 'element'] = ''
        self.df.at[tag, 'key'] = key
        self.df.loc[tag, 'tag'] = ''
        self.df.loc[tag, 'lower'] = lower
        self.df.loc[tag, 'upper'] = upper
        self.df.loc[tag, 'lower_bound'] = lower_bound
        self.df.loc[tag, 'upper_bound'] = upper_bound
        self.df.loc[tag, 'distributed'] = distributed
        self.df.loc[tag, 'initial_value'] = np.nan




    def distribute(self, model):
        """
        Unpacks constraints for each parameter and element for fully distributed calibration.

        :param model: SWMM model
        :type model: swmmio.Model
        :return: Updated Constraint object
        :rtype: Constraint
        """
        
        #if row["distributed"]:
        cps = list()

        # some parameters have more than one 'key' (e.g., hydrographs have 'index' and 'Response')
        # you might have multiple unit hydrographs (UH1, UH2, etc.) and multiple responses (short, medium, long) for each hydrograph
        # in such cases, we convert the SWMM dataframe (stored at swmmio.Model.inp) to a multi-indexed dataframe
        # since we don't want to modify the Model object directly, we do this change on the fly
        # if a constraint has a multi-index (identifiable by a tuple in the 'element' attribute), we convert the SWMM dataframe to a multi-index to access the corresponding value
        if len(self.key) > 1:
            # in the case of a multi-indexed parameter, convert a copy of the SWMM section to a multi-index
            index = self.make_multi_index(model)
            swmm_df = getattr(model.inp, self.section).copy()
            swmm_df.set_index(index, inplace=True)
        else:
            index = getattr(getattr(model.inp, self.section),self.key[0]).to_list()

        for id in index:
            cp = self.copy()
            cp.element = id
            
            if isinstance(id, tuple):
                id_str = tuple_to_str(id)
            else:
                id_str = str(id)

            # create a unique 'tag' to use as the index for the constraints
            # it contains all the information needed to set the parameter in the SWMM model
            tag = "{}.{}.{}".format(cp.section, cp.attribute, id_str)
            cp.tag = tag

            if isinstance(id, tuple):
                cp.initial_value = swmm_df.loc[id, cp.attribute]
            else:
                cp.initial_value = getattr(model.inp, cp.section).loc[id, cp.attribute]
            
            if cp.initial_value != np.nan:
                cps.append(cp)
        return cps

    def set_relative_bounds(self, upper:float=0.5, lower:float=0.5):
        """
        Sets the relative bounds for the parameters.
        """
        
        self.lower = self.initial_value * (1-lower)
        self.upper = self.initial_value * (1+upper)
        return self

# TODO: fix this function

def cons_to_df(cons: list[CalParam]) -> pd.DataFrame:
    """
    Converts a list of constraints to a DataFrame.

    :param cons: List of constraints
    :type cons: list[Constraint]
    :return: DataFrame of constraints
    :rtype: pd.DataFrame
    """

    df = pd.DataFrame(index=[cons.tag for cons in cons])
    df['section'] = [cons.section for cons in cons]
    df['attribute'] = [cons.attribute for cons in cons]
    df['element'] = [cons.element for cons in cons]
    df['key'] = [cons.key for cons in cons]
    df['tag'] = [cons.tag for cons in cons]
    df['lower'] = [cons.lower for cons in cons]
    df['upper'] = [cons.upper for cons in cons]
    df['lower_bound'] = [cons.lower_bound for cons in cons]
    df['upper_bound'] = [cons.upper_bound for cons in cons]
    df['distributed'] = [cons.distributed for cons in cons]
    df['initial_value'] = [cons.initial_value for cons in cons]
    df['parent_section'] = [cons.parent_section for cons in cons]
    df['level'] = 0
    df['node'] = ''
    
    return df


def get_calibration_order(cons:list[CalParam], model:swmmio.Model) -> list[CalParam]:
    """
    Gets the calibration order based on the number of upstream nodes.

    :param model: SWMM model
    :type model: swmmio.Model
    :return: Calibration order and node order
    :rtype: list[str]
    """

    

    node_level = {node:len(get_upstream_nodes(model.network, node)) for node in model.network.nodes}

    cons_out = []


    """
    assign a level to each calibration parameter based on the number of upstream nodes
    levels increase as you move downstream
    nodes with the same level are incremented by a small amount to avoid overlapping constraints
    """
    for c in cons:

        if c.parent_section == "subcatchments":
            node = model.inp.subcatchments['Outlet'].to_dict()[c.element]
            level = node_level[node]

        elif c.parent_section in ["conduit", "conduits", "link","links"]:
            node = model.inp.conduits['InletNode'].to_dict()[c.element]
            level = node_level[node]

        elif c.parent_section in ["nodes","node"]:
            node = c.element
            level = node_level[node]

        else:
            # some calibration parameters are not associated with a specific node, so these last
            level = 0 #np.max(np.array(list(node_level.values()))) + 100
            node = "all"

        c.level = level
        c.node = node
        cons_out.append(c)

    cons = deepcopy(cons_out)

    
    nodes = np.unique([c.node for c in cons if c.node != "None"]).tolist()

    level_increment = 1/(len(nodes)+1)
    level_adjusments = {node:ii * level_increment for ii, node in enumerate(nodes)}
    
    cons_out = []
    for c in cons:
        if c.node != "None":
            c.level = c.level + level_adjusments[c.node]
        cons_out.append(c)


        

    return cons_out


# Set parameters constraints for calibration optimization
def get_cal_params(routine, model):
    """
    Get the hard-coded constraints according to the specific calibration routine.

    :param routine: Calibration routine (e.g., 'dry', 'wet', 'tss')
    :type routine: str
    :param model: SWMM model
    :type model: swmmio.Model
    :return: Calibration constraints
    :rtype: CalParam
    """
    cps = list()
    if routine == "dry":
        cps.append(CalParam(section='dwf', attribute='AverageValue', lower=1, upper=1, lower_bound=0.0, upper_bound=10**6, distributed=True))
        cps.append(CalParam(section='inflows', attribute='Baseline', lower=1, upper=1, lower_bound=0.0, upper_bound=10**6, distributed=True))
        cps.append(CalParam(section='xsections', attribute='Geom1', lower=1, upper=1, lower_bound=0.01, upper_bound=10, distributed=True))
        #cps.append(CalParam(section='conduits', attribute='Length', lower=1, upper=10, lower_bound=0.01, upper_bound=100, distributed=True))
        #cps.append(CalParam(section='conduits', attribute='Roughness', lower=1, upper=5, lower_bound=1e-6, upper_bound=10, distributed=True))
        cps.append(CalParam(section='conduits', attribute='MaxFlow', lower=1, upper=2, lower_bound=0, upper_bound=10, distributed=True))

    if routine == "wet":
        #cps.append(CalParam(section='infiltration', attribute='CurveNum', lower=1, upper=1, lower_bound=0, upper_bound=100, distributed=True))
        cps.append(CalParam(section='subcatchments', attribute='Width', lower=0.5, upper=0.5, lower_bound=0.1, upper_bound=10**6, distributed=True))
        #cps.append(CalParam(section='subcatchments', attribute='PercSlope', lower=1, upper=1, lower_bound=0, upper_bound=100, distributed=True))
        cps.append(CalParam(section='subcatchments', attribute='Area', lower=1, upper=0, lower_bound=0, upper_bound=1E6, distributed=True))
        cps.append(CalParam(section='rdii', attribute='SewerArea', lower=1, upper=0.5, lower_bound=0, upper_bound=1E10, distributed=True))
        cps.append(CalParam(section='hydrographs', attribute='R', lower=0.5, upper=0.5, lower_bound=0, upper_bound=0.33, distributed=False, key=["index","Response"]))
        cps.append(CalParam(section='hydrographs', attribute='T', lower=0.5, upper=0.5, lower_bound=0, upper_bound= 7*24, distributed=False, key=["index","Response"]))
        cps.append(CalParam(section='hydrographs', attribute='K', lower=0.5, upper=0.5, lower_bound=0, upper_bound=7*24, distributed=False, key=["index","Response"]))

    if routine == "tss":
        cps.append(CalParam(section='pollutants', attribute='InitConcen', lower=-1, upper=1, lower_bound=0, upper_bound=1e3, distributed=False))
        cps.append(CalParam(section='pollutants', attribute='DWFConcen', lower=-1, upper=1, lower_bound=0, upper_bound=1e3, distributed=False))
        cps.append(CalParam(section='buildup', attribute='Coeff1', lower=-1, upper=2, lower_bound=0, upper_bound=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='buildup', attribute='Coeff2', lower=-1, upper=2, lower_bound=0, upper_bound=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='buildup', attribute='Coeff3', lower=-1, upper=2, lower_bound=0, upper_bound=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='washoff', attribute='Coeff2', lower=-1, upper=2, lower_bound=0, upper_bound=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='washoff', attribute='Coeff2', lower=-1, upper=2, lower_bound=0, upper_bound=1e2, distributed=False, key=["index","Pollutant"]))


    cps_distributed = []
    for cp in cps:
        cps_distributed += cp.distribute(model)

    # update the bounds relative to the initial values of each calibration parameter
    cps_distributed = [cp.set_relative_bounds(upper=cp.upper, lower=cp.lower) for cp in cps_distributed]

    #if routine == "wet":
    #    for cp in cps_distributed:
    #    # remove subcatchments with zero area
    #        if cp.distributed and cp.section == "subcatchments":
    #            cps_distributed.remove(cp)
            
    return cps_distributed