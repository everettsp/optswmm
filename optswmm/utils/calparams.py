"""SWMM Calibration Parameters Module"""

import pandas as pd
import numpy as np
from copy import deepcopy
import swmmio
from pathlib import Path
from swmmio import Model
from warnings import warn as warning

from pyswmm import Simulation, SimulationPreConfig, Subcatchments, Nodes, Links

from optswmm.defs import SWMM_SECTION_SUBTYPES, ROW_ATTRIBUTES, ROW_INDICES, PARAM_INDICES
from optswmm.utils.networkutils import get_upstream_nodes


from typing import Optional

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
                 element: str = '',
                 key:list[str]=["index"], 
                 lower: float = 0.0, 
                 upper: float = np.inf, 
                 lower_limit: float = -np.inf, 
                 upper_limit: float = np.inf, 
                 distributed: bool = False,
                 relative: bool = False,
                 relative_bounds: bool = True,
                 col_index: int|None = None,
                 initial_value:float = np.nan,
                 opt_value:float = np.nan,
                 opt_value_absolute:float = np.nan,
                 row_attribute: Optional[str] = None,
                 row_index: int = 0,
                 level: int = 0,
                 ii: int = 0,
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
        self.relative_bounds = relative_bounds
        self.initial_value = initial_value
        self.x0 = float('nan')
        self.col_index = col_index  # numeric integer used while setting sim preconfig
        self.row_index = row_index
        self.row_attribute = row_attribute # used for hydrographs, e.g., 'Short', 'Medium', 'Long'
        self.opt_value = opt_value # value after optimization
        self.opt_value_absolute = opt_value_absolute # absolute value after optimization
        self.level = level # used for calibration order
        self.node = None # used for calibration order
        self.tag = "{}.{}.{}".format(self.section, self.attribute, self.element) # unique tag for the parameter
        self.ii = ii  # index for mapping to cal results; set in get_initial_values
        # if self.row_attribute is not None:
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

        self.col_index = PARAM_INDICES[self.section][self.attribute]
        
        if self.row_attribute is not None:
            if self.row_attribute not in ROW_INDICES.get(self.section, {}):
                raise ValueError(f"row attribute '{self.row_attribute}' not found in section '{self.section}'. Available attributes: {list(ROW_INDICES.get(self.section, {}).keys())}")
            self.row_index = ROW_INDICES.get(self.section, {}).get(self.row_attribute, None)

        #if not isinstance(self.distributed, bool):
        #    raise ValueError(f"constraints 'distributed' parameter must be of type bool")

    def make_multi_index(self, model) -> pd.MultiIndex:
        index = [getattr(getattr(model.inp, self.section),x).to_list() for x in self.key]
        index = pd.MultiIndex.from_arrays(index)
        return index

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

        # some parameters have more than one 'key' (e.g., hydrographs have 'col_index' and 'Response')
        # you might have multiple unit hydrographs (UH1, UH2, etc.) and multiple responses (short, medium, long) for each hydrograph
        # in such cases, we convert the SWMM dataframe (stored at swmmio.Model.inp) to a multi-indexed dataframe
        # since we don't want to modify the Model object directly, we do this change on the fly
        # if a constraint has a multi-index (identifiable by a tuple in the 'element' attribute), we convert the SWMM dataframe to a multi-index to access the corresponding value
        
        
        if len(self.key) > 1:
            # in the case of a multi-indexed parameter, convert a copy of the SWMM section to a multi-index
            element_list = self.make_multi_index(model)
            #swmm_df = getattr(model.inp, self.section).copy()
            #swmm_df.set_index(index, inplace=True)
        else:
            element_list = getattr(getattr(model.inp, self.section),self.key[0]).to_list()

        element_list = np.unique(element_list).tolist()

        # remove outfalls from calibration elements
        outfalls = getattr(model.inp, "outfalls").index.to_list()
        element_list = [id for id in element_list if id not in outfalls]

        for id in element_list:
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

    def relative_bounds_to_absolute(self, upper:float|None=None, lower:float|None=None):
        """
        if you want to adjust absolute parameters, 
        but specify relative bounds (e.g., 10% above and below the initial value),
        this function will convert the bounds for a calibration parameter.
        """
        if self.relative_bounds:
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

            if cp.lower > cp.upper:
                raise ValueError(f"Lower bound {cp.lower} is greater than upper bound {cp.upper} for parameter {cp.section}.{cp.attribute}.")
            
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
                    # NOTE: multi-indexing is currently not supported
                    # in the case of a multi-indexed parameter, convert a copy of the SWMM section to a multi-index
                    index = cp.make_multi_index(model)
                    swmm_df = getattr(model.inp, cp.section).copy()
                    swmm_df.set_index(index, inplace=True)
                    cp.initial_value = swmm_df.loc[cp.element, cp.attribute]

                else:
                    index = getattr (getattr(model.inp, cp.section),cp.key[0]).to_list()
                    cp.initial_value = getattr(model.inp, cp.section).loc[cp.element, cp.attribute]

                    if np.array(cp.initial_value).size > 1:
                        cp.initial_value = cp.initial_value[cp.row_index]
                    


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


    def relative_bounds_to_absolute(self, upper: float | None = None, lower: float | None = None):
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

    def to_df(self) -> pd.DataFrame:
        """
        Convert the list of CalParams to a pandas DataFrame.

        :return: DataFrame representation of CalParams
        :rtype: pd.DataFrame
        """
        return _cons_to_df(self)

    def from_df(self, filename: Path) -> "CalParams":
        """
        Convert a DataFrame to a list of CalParams.

        :param filename: Path to CSV file to read
        :type filename: Path
        :return: List of CalParams
        :rtype: CalParams
        """
        df = pd.read_csv(filename, index_col=0)
        df = df.replace({np.nan: None})
        
        cps = []
        for _, row in df.iterrows():
            cp = CalParam(
                section=row['section'],
                attribute=row['attribute'],
                element=row['element'],
                key=eval(row['key']) if isinstance(row['key'], str) else row['key'],
                lower=row['lower'],
                upper=row['upper'],
                lower_limit=row['lower_limit'],
                upper_limit=row['upper_limit'],
                distributed=row['distributed'],
                relative=row['relative'],
                relative_bounds=row['relative_bounds'],
                col_index=row['col_index'],
                initial_value=row['initial_value'],
                opt_value=row.get('opt_value', np.nan),
                opt_value_absolute=row.get('opt_value_absolute', np.nan),
                row_attribute=row.get('row_attribute', ''),
            )
            cps.append(cp)
        return CalParams(cps)

    def set_values(self, values):
        """
        Set the optimization values for each calibration parameter.

        :param values: List of optimization values.
        :type values: list[float]
        """
        if len(values) != len(self):
            raise ValueError("Length of values must match length of calibration parameters.")

        for cp, val in zip(self, values):
            cp.opt_value = val
            if cp.relative:
                cp.opt_value_absolute = cp.initial_value * (1 + val)
            else:
                cp.opt_value_absolute = val

    def make_simulation_preconfig(self, model):
        """
        Set up SimulationPreConfig with parameter updates.
        
        :param cal_params: List of calibration parameters.
        :type cal_params: list[CalParam]
        :param values: Parameter values from optimization.
        :type values: list[float]
        :param opt_config: Optimization configuration.
        :type opt_config: OptConfig
        :returns: Configured SimulationPreConfig object
        :rtype: SimulationPreConfig
        """
        spc = SimulationPreConfig()


        for cp in self:
            val = cp.opt_value_absolute
            
            # Handle non-distributed parameters
            if not cp.distributed:
                if not cp.relative:
                    raise NotImplementedError("Non-distributed calibration params must be set to 'relative'")

                # Get the section dataframe and element IDs
                section_df = getattr(model.inp, cp.section)
                element_ids = section_df.index.unique()

                for element_id in element_ids:
                    """
                    # Handle multi-index sections (e.g., hydrographs, LIDs, etc.)
                    if len(section_df.index.unique()) != len(section_df):
                        subsec = section_df.loc[[element_id], :]
                        model_val = subsec.iloc[cp.row_index, :][cp.attribute]
                    # Handle single-index sections
                    else:
                        model_val = section_df.loc[element_id, cp.attribute]
                    """
                    # Apply physical constraints
                    new_val = cp.truncate(new_val)

                    # Add update to simulation preconfig
                    spc.add_update_by_token(
                        section=cp.section,
                        obj_id=element_id,
                        index=cp.col_index,
                        new_val=new_val,
                        row_num=cp.row_index
                    )

            # Handle distributed parameters
            else:
                # Calculate new value (relative or absolute)
                if cp.relative:
                    new_val = cp.initial_value * (1 + val)
                else:
                    new_val = val

                # Apply physical constraints
                new_val = cp.truncate(new_val)

                # Add update to simulation preconfig
                spc.add_update_by_token(
                    section=cp.section,
                    obj_id=cp.element,
                    index=cp.col_index,
                    new_val=new_val,
                    row_num=cp.row_index
                )


        return spc
        
def _cons_to_df(cons: CalParams) -> pd.DataFrame:

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
    df['col_index'] = [cons.col_index for cons in cons]
    df['lower'] = [cons.lower for cons in cons]
    df['upper'] = [cons.upper for cons in cons]
    df['lower_limit'] = [cons.lower_limit for cons in cons]
    df['upper_limit'] = [cons.upper_limit for cons in cons]
    df['distributed'] = [cons.distributed for cons in cons]
    df['relative'] = [cons.relative for cons in cons]
    df['relative_bounds'] = [cons.relative_bounds for cons in cons]
    df['initial_value'] = [cons.initial_value for cons in cons]
    df['x0'] = [cons.x0 for cons in cons]
    df['row_index'] = [cons.row_index for cons in cons]
    df['row_attribute'] = [cons.row_attribute for cons in cons]
    df['parent_section'] = [cons.parent_section for cons in cons]
    df['tag'] = [cons.tag for cons in cons]
    # Optionally add 'level' and 'node' if present
    df['level'] = [getattr(cons, 'level', None) for cons in cons]
    df['node'] = [getattr(cons, 'node', None) for cons in cons]
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
    :rtype: CalParams
    """
    cps = list()
    if routine == "dry":
        cps.append(CalParam(section='dwf', attribute='AverageValue', lower=1, upper=1, lower_limit=0.0, upper_limit=10**6, distributed=True))
        cps.append(CalParam(section='inflows', attribute='Baseline', lower=1, upper=1, lower_limit=0.0, upper_limit=10**6, distributed=True))
        cps.append(CalParam(section='xsections', attribute='Geom1', lower=1, upper=1, lower_limit=0.01, upper_limit=10, distributed=True))
        cps.append(CalParam(section='conduits', attribute='MaxFlow', lower=1, upper=2, lower_limit=0, upper_limit=10, distributed=True))

    if routine == "wet":
        cps.append(CalParam(section='infiltration', attribute='CurveNum', lower=1, upper=1, lower_limit=0, upper_limit=100, distributed=True))
        cps.append(CalParam(section='subcatchments', attribute='Width', lower=0.5, upper=0.5, lower_limit=0.1, upper_limit=10**6, distributed=True))
        cps.append(CalParam(section='subcatchments', attribute='Area', lower=1, upper=0, lower_limit=0, upper_limit=1E6, distributed=True))

    if routine == "tss":
        cps.append(CalParam(section='pollutants', attribute='InitConcen', lower=-1, upper=1, lower_limit=0, upper_limit=1e3, distributed=False))
        cps.append(CalParam(section='pollutants', attribute='DWFConcen', lower=-1, upper=1, lower_limit=0, upper_limit=1e3, distributed=False))
        cps.append(CalParam(section='buildup', attribute='Coeff1', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["col_index","Pollutant"]))
        cps.append(CalParam(section='buildup', attribute='Coeff2', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["col_index","Pollutant"]))
        cps.append(CalParam(section='buildup', attribute='Coeff3', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["col_index","Pollutant"]))
        cps.append(CalParam(section='washoff', attribute='Coeff2', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["col_index","Pollutant"]))
        cps.append(CalParam(section='washoff', attribute='Coeff2', lower=-1, upper=2, lower_limit=0, upper_limit=1e2, distributed=False, key=["col_index","Pollutant"]))


    cps_obj = CalParams(cps)
    cps_distributed = cps_obj.distribute(model)

    # update the bounds relative to the initial values of each calibration parameter
    cps_distributed = cps_distributed.relative_bounds_to_absolute()

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
                    spc.add_update_by_token(section=cp.section, obj_id=element_id, col_index=cp.col_index, new_val=new_val)
        
            # if it's a distributed calibration parameter
        else:
            # if the calibrated value 'val' is the relative change - so we need to calculate the new model values relative to the initial values
            if cp.relative:
                new_val = cp.initial_value * (1 + val)
            # otherwise, the value can be set directly
            else:
                new_val = val
                
            new_val = cp.truncate(new_val)
            spc.add_update_by_token(section=cp.section, obj_id=cp.element, col_index=cp.col_index, new_val=new_val)
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
                col_index=row.col_index,
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