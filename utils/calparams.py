"""TEST"""

import pandas as pd
import numpy as np
from copy import deepcopy
from defs import SWMM_SECTION_SUBTYPES
from utils.networkutils import get_upstream_nodes
import swmmio

from pathlib import Path
from swmmio import Model
from warnings import warn as warning

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
    },
    "conduits":{
        "FromNode":1,
        "ToNode":2,
        "Length":3,
        "Roughness":4
    },
    "xsections":{
        "Shape":1,
        "Geom1":2,
        "Geom2":3,
    },
    "inflows":{
        "Mfactor":4,
        "Sfactor":5,
        "Baseline":6,
        "Pattern":7,
    }
}


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
                 attribute: str,
                 element: str = None,
                 key:list[str]=["index"], 
                 lower: float = 0.0, 
                 upper: float = np.inf, 
                 lower_limit: float = -np.inf, 
                 upper_limit: float = np.inf, 
                 distributed: bool = False,
                 relative: bool = False,
                 index: int = None,
                 initial_value:float = None,
                 ):
        
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
        :param lower_limit: Physics-based lower constraint for value (e.g., inflow >= 0)
        :type lower_limit: float
        :param upper_limit: Physics-based upper constraint for value
        :type upper_limit: float
        :raises NotImplementedError: If section is not found in SWMM_SECTION_SUBTYPES
        """

        if type(key) is str:
            key = [key]
            
        self.tag = '' # depreciated
        self.section = section
        self.attribute = attribute
        self.element = element
        self.key = key # for multi-indexed parameters (made ~irrelevant by sim preconfig)
        self.parent_section = ''
        self.ii = None  # index for mapping to cal results; set in get_initial_values
        self.lower = lower # lower search
        self.upper = upper # upper search
        self.lower_limit = lower_limit # lower physical constraint
        self.upper_limit = upper_limit # upper physical constraint
        self.distributed = distributed # whether to distribute the parameter across all elements
        self.relative = relative # relative change (e.g., 0.5 is +50% the initial value), in contrast to absolute (e.g., +5mm/hr)
        self.initial_value = initial_value
        self.x0 = None
        self.index = index  # numeric integer used while setting sim preconfig
        self.row_num = None # depreciated
        #self.distributed = False
        self._standardize()

    def copy(self):
        """
        Returns a deep copy of Constraint.

        :return: Deep copy of Constraint
        :rtype: Constraint
        """
        return deepcopy(self)

    def truncate(self, val):
        """
        Truncates the search bounds for distributed parameters.
        """
        if self.lower_limit > val:
            val = self.lower_limit
        if self.upper_limit < val:
            val = self.upper_limit
        return val
    
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


        self.index = PARAM_INDICES[self.section][self.attribute]



        #if not isinstance(self.distributed, bool):
        #    raise ValueError(f"constraints 'distributed' parameter must be of type bool")

    def make_multi_index(self, model) -> pd.MultiIndex:
        index = [getattr(getattr(model.inp, self.section),x).to_list() for x in self.key]
        index = pd.MultiIndex.from_arrays(index)
        return index

    def append_parameter_constraint(self, section: str, attribute: str, key:list[str]=["index"], lower: float = 0.0, upper: float = np.inf, lower_limit: float = -np.inf, upper_limit: float = np.inf, distributed: bool=False):
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
        :param lower_limit: Physics-based lower constraint for value (e.g., inflow >= 0)
        :type lower_limit: float
        :param upper_limit: Physics-based upper constraint for value
        :type upper_limit: float
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
        self.df.loc[tag, 'lower_limit'] = lower_limit
        self.df.loc[tag, 'upper_limit'] = upper_limit
        self.df.loc[tag, 'distributed'] = distributed
        self.df.loc[tag, 'initial_value'] = np.nan


    def distribute(self, model: swmmio.Model) -> list["CalParam"]:
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

            cps.append(cp)
        return CalParams(cps)

    def relative_bounds_to_absolute(self, upper:float=None, lower:float=None):
        """
        If not relative param changes, convert the upper and lower bounds
        from relative to absolute, using the initial values of each parameter
        """
        if not self.relative:
            if upper is None:
                upper = self.upper

            if lower is None:
                lower = self.lower

            self.lower = self.initial_value * (1+lower)
            self.upper = self.initial_value * (1+upper)

            # truncate the search bounds for distributed parameters
            if self.lower < self.lower_limit:
                self.lower = self.lower_limit
            if self.upper > self.upper_limit:
                self.upper = self.upper_limit
        return self


class CalParams(list[CalParam]):
    def __init__(self, *args):
        """
        Initialize a list of CalParam objects.
        """
        super().__init__(*args)

    def append(self, cal_param: CalParam):
        """
        Append a CalParam object to the list.

        :param cal_param: Calibration parameter to append
        :type cal_param: CalParam
        """
        if not isinstance(cal_param, CalParam):
            raise TypeError("Only CalParam objects can be appended.")
        super().append(cal_param)


    def get_bounds(self):
        bounds = []

        for cp in self:
            
            # NOTE: temporarily commented out the truncation of the bounds, this is done later instead

            #if cp.lower < cp.lower_limit:
            #    lower = cp.lower_limit
            #else:
            #    lower = cp.lower
            #if cp.upper > cp.upper_limit:
            #    upper = cp.upper_limit
            #else:
            #    upper = cp.upper

            bounds.append((cp.lower, cp.upper))
        return bounds

    def filter_by_section(self, section: str) -> "CalParams":
        """
        Filter the list of CalParams by section.

        :param section: Section to filter by
        :type section: str
        :return: Filtered CalParams
        :rtype: CalParams
        """
        return CalParams([cp for cp in self if cp.section == section])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the list of CalParams to a pandas DataFrame.

        :return: DataFrame representation of CalParams
        :rtype: pd.DataFrame
        """
        return cons_to_df(self)

    def distribute(self, model) -> "CalParams":
        """
        Distribute all CalParams in the list.

        :param model: SWMM model
        :type model: swmmio.Model
        :return: Distributed CalParams
        :rtype: CalParams
        """
        distributed = []
        for cp in self:
            if cp.distributed:
                distributed.extend(cp.distribute(model))
            else:
                distributed.append(cp)
        return CalParams(distributed)


    def get_initial_values(self, model: swmmio.Model) -> list["CalParam"]:
        cps = []

        for cp in self:

            if not cp.distributed:                
                cp.initial_value = getattr(model.inp, cp.section).loc[:, cp.attribute].mean()

            else:
                if len(cp.key) > 1:
                    # in the case of a multi-indexed parameter, convert a copy of the SWMM section to a multi-index
                    index = cp.make_multi_index(model)
                    swmm_df = getattr(model.inp, cp.section).copy()
                    swmm_df.set_index(index, inplace=True)
                    cp.initial_value = swmm_df.loc[cp.element, cp.attribute]

                else:
                    index = getattr(getattr(model.inp, cp.section),cp.key[0]).to_list()
                    cp.initial_value = getattr(model.inp, cp.section).loc[cp.element, cp.attribute]

            if cp.relative:
                cp.x0 = -0.01
            else:
                cp.x0 = cp.initial_value * 0.9

            if cp.initial_value != np.nan:
                cps.append(cp)


        # here we set a numeric index to the cal-params to map the cal params saved during cal
        for ii in range(len(cps)):
            cps[ii].ii = ii
        return CalParams(cps)


    def relative_bounds_to_absolute(self, upper: float = None, lower: float = None):
        """
        Set relative bounds for all CalParams in the list.

        :param upper: Upper bound multiplier
        :type upper: float
        :param lower: Lower bound multiplier
        :type lower: float
        """
        cps = []
        for cp in self:
            if not cp.relative:
                cp.relative_bounds_to_absolute(upper=upper, lower=lower)
            cps.append(cp)
        return CalParams(cps)


# TODO: fix this function

def cons_to_df(cons: list[CalParam]) -> pd.DataFrame:
    """
    Converts a list of constraints to a DataFrame.

    :param cons: List of constraints
    :type cons: list[Constraint]
    :return: DataFrame of constraints
    :rtype: pd.DataFrame
    """

    df = pd.DataFrame(index=range(len(cons)))
    df['section'] = [cons.section for cons in cons]
    df['attribute'] = [cons.attribute for cons in cons]
    df['element'] = [cons.element for cons in cons]
    df['key'] = [cons.key for cons in cons]
    df['ii'] = [cons.ii for cons in cons]
    df['index'] = [cons.index for cons in cons]
    df['lower'] = [cons.lower for cons in cons]
    df['upper'] = [cons.upper for cons in cons]
    df['lower_limit'] = [cons.lower_limit for cons in cons]
    df['upper_limit'] = [cons.upper_limit for cons in cons]
    df['distributed'] = [cons.distributed for cons in cons]
    df['relative'] = [cons.relative for cons in cons]
    df['initial_value'] = [cons.initial_value for cons in cons]
    df['parent_section'] = [cons.parent_section for cons in cons]
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
        cps.append(CalParam(section='dwf', attribute='AverageValue', lower=1, upper=1, lower_limit=0.0, upper_limit=10**6, distributed=True))
        cps.append(CalParam(section='inflows', attribute='Baseline', lower=1, upper=1, lower_limit=0.0, upper_limit=10**6, distributed=True))
        cps.append(CalParam(section='xsections', attribute='Geom1', lower=1, upper=1, lower_limit=0.01, upper_limit=10, distributed=True))
        #cps.append(CalParam(section='conduits', attribute='Length', lower=1, upper=10, lower_limit=0.01, upper_limit=100, distributed=True))
        #cps.append(CalParam(section='conduits', attribute='Roughness', lower=1, upper=5, lower_limit=1e-6, upper_limit=10, distributed=True))
        cps.append(CalParam(section='conduits', attribute='MaxFlow', lower=1, upper=2, lower_limit=0, upper_limit=10, distributed=True))

    if routine == "wet":
        cps.append(CalParam(section='infiltration', attribute='CurveNum', lower=1, upper=1, lower_limit=0, upper_limit=100, distributed=True))
        cps.append(CalParam(section='subcatchments', attribute='Width', lower=0.5, upper=0.5, lower_limit=0.1, upper_limit=10**6, distributed=True))
        #cps.append(CalParam(section='subcatchments', attribute='PercSlope', lower=1, upper=1, lower_limit=0, upper_limit=100, distributed=True))
        cps.append(CalParam(section='subcatchments', attribute='Area', lower=1, upper=0, lower_limit=0, upper_limit=1E6, distributed=True))

    if routine == "tss":
        cps.append(CalParam(section='pollutants', attribute='InitConcen', lower=-1, upper=1, lower_limit=0, upper_limit=1e3, distributed=False))
        cps.append(CalParam(section='pollutants', attribute='DWFConcen', lower=-1, upper=1, lower_limit=0, upper_limit=1e3, distributed=False))
        cps.append(CalParam(section='buildup', attribute='Coeff1', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='buildup', attribute='Coeff2', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='buildup', attribute='Coeff3', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='washoff', attribute='Coeff2', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["index","Pollutant"]))
        cps.append(CalParam(section='washoff', attribute='Coeff2', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["index","Pollutant"]))


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


def create_simpreconfig(cps, vals, model_file):

    if isinstance(model_file, Path):
        model_file = str(model_file)

    if len(cps) != len(vals):
        raise ValueError("Length of cps and vals must match")

    model = Model(model_file)
    spc = SimulationPreConfig()

    for cp, val in zip(cps, vals):
        # if it's not a distributed parameter, get all element ids and apply the new value everywhere
        if not cp.distributed:
            with Simulation(model_file) as sim:
                if not cp.relative:
                    raise NotImplementedError("Non-distributed calibration params must be set to 'relative'")
                
                # the calibrated value 'val' is the relative change - so we need to calculate the new model values relative to the initial values
                val_map = getattr(Model(model_file).inp, cp.section).loc[:,cp.attribute].to_dict()
                for element_id, model_val in val_map.items():
                    new_val = model_val * (1 + val)
                    new_val = cp.truncate(new_val)
                    spc.add_update_by_token(section=cp.section, obj_id=element_id, index=cp.index, new_val=new_val)
        
            # if it's a distributed calibration parameter
        else:
            # if the calibrated value 'val' is the relative change - so we need to calculate the new model values relative to the initial values
            if cp.relative:
                new_val = cp.initial_value * (1 + val)
            # otherwise, the value can be set directly
            else:
                new_val = val
                
            new_val = cp.truncate(new_val)
            spc.add_update_by_token(section=cp.section, obj_id=cp.element, index=cp.index, new_val=new_val)
    return spc



def get_simpreconfig_at_iter(run_dir, iter=None):
    # load the calibration parameter values
    param_file = run_dir / "results_params.txt"
    params_df = pd.read_csv(param_file, sep=",", index_col=0)  # Adjust the separator if necessary

    # load the calibration parameter metadata
    cal_params = pd.read_csv(run_dir / "calibration_parameters.csv")
    df = cal_params.merge(params_df, on="ii", how="inner")

    if iter is None:
        iter = int(df.iter.max())
        warning("iter not specified, using last iter")


    if iter not in df.iter.unique():
        iter = df.iter.sub(iter).abs().idxmin()
        warning(f"iter {iter} not found, using closest iter {df.iter[iter]} instead")

    # filter the dataframe to only include the specified iteration
    df = df[df.iter == iter]



    # convert dataframe results to CalParams object
    cps = []
    for name, row in df.iterrows():
        cp = CalParam(section=row.section,
                attribute=row.attribute,
                element=row.element,
                index=row.index,
                lower=row.lower,
                upper=row.upper,
                upper_limit=row.upper_limit,
                lower_limit=row.lower_limit,
                relative=row.relative,
                distributed=row.distributed,
                initial_value = row.initial_value)
        cps.append(cp)
    cps = CalParams(cps)
    model_file = str(run_dir.parent.parent / f"{run_dir.parent.parent.stem}_cal.inp")

    spc=create_simpreconfig(cps=cps, vals=df.cal_val, model_file=model_file)

    return spc