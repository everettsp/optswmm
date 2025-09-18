"""SWMM Calibration Parameters Module"""

import pandas as pd
import numpy as np
from copy import deepcopy
import swmmio
from pathlib import Path
from swmmio import Model
from warnings import warn as warning
from typing import Optional, Union, List

from pyswmm import SimulationPreConfig, Simulation

from optswmm.defs import SWMM_SECTION_SUBTYPES, ROW_INDICES, PARAM_INDICES
from optswmm.utils.networkutils import get_upstream_nodes


def tuple_to_str(tup: tuple) -> str:
    """
    Converts a tuple to a string.

    :param tup: Tuple to convert
    :type tup: tuple
    :return: String
    :rtype: str
    """
    return '_'.join([str(x) for x in tup])


class CalParam:
    def __init__(self,
                 section: str, 
                 attribute: str,
                 element: str = '',
                 key: Union[str, List[str]] = "index", 
                 lower_rel: Optional[float] = None, 
                 upper_rel: Optional[float] = None,
                 lower_abs: Optional[float] = None, 
                 upper_abs: Optional[float] = None,
                 lower_limit_abs: float = -np.inf,
                 upper_limit_abs: float = np.inf,
                 distributed: bool = False,
                 mode: str = 'direct',
                 relative_bounds: bool = True,
                 col_index: Optional[int] = None,
                 initial_value: float = np.nan,
                 opt_value: float = np.nan,
                 model_value: float = np.nan,
                 row_attribute: Optional[str] = None,
                 row_index: int = 0,
                 level: int = 0,
                 ii: int = 0,
                 active: bool = True,
                 set_model_value = None
                 ):
        """
        Initialize a parameter for SWMM calibration.

        :param section: Section of SWMM INP file (e.g., subcatchments)
        :param attribute: Attribute of the section (e.g., slope)
        :param element: Specific element ID (for distributed parameters)
        :param key: Key for accessing multi-indexed parameters
        :param lower: Lower constraint of search-space
        :param upper: Upper constraint of search-space
        :param lower_limit: Physics-based lower constraint for value
        :param upper_limit: Physics-based upper constraint for value
        :param distributed: Whether to distribute parameter across all elements
        :param multiplicative: Whether changes are multiplicative (multiplier) or direct
        :param relative_bounds: Whether bounds are specified multiplicative to initial value
        :param col_index: Column index for SWMM section
        :param initial_value: Initial parameter value from model
        :param opt_value: Optimized parameter value
        :param opt_value_absolute: Absolute optimized parameter value
        :param row_attribute: Row attribute for multi-row sections
        :param row_index: Row index for multi-row sections
        :param level: Calibration level for hierarchical optimization
        :param ii: Index for mapping to calibration results
        :param active: Whether parameter is active in calibration. Toggled during 'hierarchical' calibration method
        :raises NotImplementedError: If section is not found in SWMM_SECTION_SUBTYPES
        """
        if isinstance(key, str):
            key = [key]
            
        self.section = section
        self.attribute = attribute
        self.element = element
        self.key = key
        self.lower_abs = lower_abs
        self.upper_abs = upper_abs
        self.lower_rel = lower_rel
        self.upper_rel = upper_rel
        self.lower_limit_abs = lower_limit_abs
        self.upper_limit_abs = upper_limit_abs
        self.distributed = distributed
        self.mode = mode
        self.relative_bounds = relative_bounds
        self.initial_value = initial_value
        self.opt_value = opt_value
        self.model_value = model_value
        self.row_attribute = row_attribute
        self.row_index = row_index
        self.level = level
        self.ii = ii
        self.active = active
        # Set derived attributes
        self.x0 = float('nan')
        self.col_index = col_index
        self.parent_section = ''
        self.node = ''
        self.tag = f"{self.section}.{self.attribute}.{self.element}"
        self.set_model_value = set_model_value

        self._standardize()

    def copy(self) -> "CalParam":
        """Returns a deep copy of CalParam."""
        return deepcopy(self)

    def truncate(self, val: float) -> float:
        """Truncate value to within physical limits."""
        return np.clip(val, self.lower_limit_abs, self.upper_limit_abs)

    def _standardize(self):
        """Standardize and validate the calibration parameter."""
        if self.section not in SWMM_SECTION_SUBTYPES:
            raise NotImplementedError(
                f"section '{self.section}' not found in SWMM_SECTION_SUBTYPES. "
                f"Available: {list(SWMM_SECTION_SUBTYPES.keys())}"
            )
        
        self.parent_section = SWMM_SECTION_SUBTYPES[self.section]
        self.col_index = PARAM_INDICES[self.section][self.attribute]
        
        if self.row_attribute is not None:
            if self.row_attribute not in ROW_INDICES.get(self.section, {}):
                raise ValueError(
                    f"row attribute '{self.row_attribute}' not found in section '{self.section}'. "
                    f"Available: {list(ROW_INDICES.get(self.section, {}).keys())}"
                )
            self.row_index = ROW_INDICES[self.section][self.row_attribute]

        # Set set_model_value function based on mode
        MODES = ["multiplicative", "additive", "direct", "linear"]
        if self.mode not in MODES:
            warning(f"Mode '{self.mode}' not recognized. Setting to 'direct'.")
            self.mode = "direct"

        if self.mode == "multiplicative":
            self.set_model_value = lambda initial_value, cal_value: initial_value * (1 + cal_value)
        elif self.mode == "additive":
            self.set_model_value = lambda initial_value, cal_value: initial_value + cal_value
        elif self.mode == "direct":
            self.set_model_value = lambda initial_value, cal_value: cal_value
        elif self.mode == "linear":
            self.set_model_value = lambda initial_value, cal_value: cal_value[1] * initial_value + cal_value[0]
        else:
            self.set_model_value = lambda val: np.nan

    def calculate_new_val(self, val):
                    if self.mode == "multiplicative":
                        return self.initial_value * (1 + val)
                    elif self.mode == "additive":
                        return self.initial_value + val
                    elif self.mode == "linear":
                        if len(val) != 2:
                            raise ValueError("Linear mode requires a tuple of (slope, intercept).")
                        return val[1] * self.initial_value + val[0]
                    else:
                        raise NotImplementedError(f"Mode '{self.mode}' is not supported.")


    def make_multi_index(self, model: swmmio.Model) -> pd.MultiIndex:
        """Create multi-index for multi-keyed parameters."""
        index_arrays = [getattr(getattr(model.inp, self.section), key).to_list() for key in self.key]
        return pd.MultiIndex.from_arrays(index_arrays)

    def distribute(self, model: swmmio.Model) -> "CalParams":
        """
        Create distributed calibration parameters for each element.

        :param model: SWMM model
        :return: CalParams object with distributed parameters
        """
        cps = []

        if len(self.key) > 1:
            element_list = self.make_multi_index(model)
        else:
            element_list = getattr(getattr(model.inp, self.section), self.key[0]).to_list()

        element_list = np.unique(element_list).tolist()

        # Remove outfalls from calibration elements
        outfalls = model.inp.outfalls.index.to_list()
        element_list = [id for id in element_list if id not in outfalls]

        for element_id in element_list:
            cp = self.copy()
            cp.element = element_id
            
            id_str = tuple_to_str(element_id) if isinstance(element_id, tuple) else str(element_id)
            cp.tag = f"{cp.section}.{cp.attribute}.{id_str}"
            cps.append(cp)
            
        return CalParams(cps)

    def calculate_bounds(self):
        if self.upper_abs is None or self.lower_abs is None:
            self = self.rel_to_abs_bounds()
        if self.upper_rel is None or self.lower_rel is None:
            self = self.abs_to_rel_bounds()
        return self
    
    def rel_to_abs_bounds(self, upper: Optional[float] = None, lower: Optional[float] = None) -> "CalParam":
        """Convert multiplicative bounds to absolute bounds based on initial value."""
        if self.relative_bounds:
            lower = self.lower_rel if lower is None else lower
            upper = self.upper_rel if upper is None else upper
            self.lower_abs = max(self.initial_value * (1 + lower), self.lower_limit_abs)
            self.upper_abs = min(self.initial_value * (1 + upper), self.upper_limit_abs)
        return self

    def abs_to_rel_bounds(self, upper: Optional[float] = None, lower: Optional[float] = None) -> "CalParam":
        """Convert absolute bounds to multiplicative bounds based on initial value."""
        if self.relative_bounds:
            upper = self.upper_abs if upper is None else upper
            lower = self.lower_abs if lower is None else lower
            self.lower_rel = lower / self.initial_value - 1
            self.upper_rel = upper / self.initial_value - 1
        return self

class CalParams(list):
    """List of CalParam objects with additional functionality."""

    def __init__(self, cal_params: Optional[List[CalParam]] = None):
        """Initialize CalParams list."""
        super().__init__(cal_params or [])

    def append(self, cal_param: CalParam):
        """Append a CalParam object to the list."""
        if not isinstance(cal_param, CalParam):
            raise TypeError("Only CalParam objects can be appended.")
        super().append(cal_param)

    def get_bounds(self) -> List[tuple]:
        """Get optimization bounds for all parameters."""
        bounds = []
        for cp in self:
            if cp.active:
                if cp.mode == "linear":
                    # Linear mode has two parameters (slope, intercept)
                    if cp.lower_abs > cp.upper_abs:
                        raise ValueError(
                            f"Lower bound {cp.lower_abs} > upper bound {cp.upper_abs} "
                            f"for parameter {cp.section}.{cp.attribute}"
                        )
                    bounds.append([(cp.lower_abs, cp.upper_abs), (cp.lower_rel, cp.upper_rel)])
                else:
                    if cp.lower_abs > cp.upper_abs:
                        raise ValueError(
                            f"Lower bound {cp.lower_abs} > upper bound {cp.upper_abs} "
                            f"for parameter {cp.section}.{cp.attribute}"
                        )
                    bounds.append((cp.lower_abs, cp.upper_abs))

        bounds = self.flatten_bounds(bounds)
        return bounds

    def flatten_bounds(self, bounds: List[Union[tuple, List[tuple]]]) -> List[tuple]:
        """
        Flatten a list of bounds that may contain tuples or lists of tuples.
        
        For linear mode parameters, each parameter may have multiple bounds 
        (one for slope, one for intercept). This function flattens them into
        a single list of tuples.
        
        :param bounds: List of bounds, where each element is either a tuple 
                      (lower, upper) or a list of tuples [(lower1, upper1), (lower2, upper2)]
        :type bounds: List[Union[tuple, List[tuple]]]
        :return: Flattened list of bounds tuples
        :rtype: List[tuple]
        
        Example:
            Input:  [(0, 1), [(0.5, 1.5), (-1, 1)], (2, 3)]
            Output: [(0, 1), (0.5, 1.5), (-1, 1), (2, 3)]
        """
        flattened = []
        for bound in bounds:
            if isinstance(bound, list):
                # Linear mode bounds - extend with all sub-bounds
                flattened.extend(bound)
            elif isinstance(bound, tuple):
                # Regular bounds - append as single tuple
                flattened.append(bound)
            else:
                raise TypeError(f"Bounds must be tuple or list of tuples, got {type(bound)}")
        return flattened

    def unflatten_bounds(self, flattened_bounds: List[tuple]) -> List[Union[tuple, List[tuple]]]:
        """
        Unflatten a list of bounds tuples back to the original structure.
        
        This is the inverse of flatten_bounds, reconstructing the original
        bounds structure based on the parameter modes.
        
        :param flattened_bounds: Flattened list of bounds tuples
        :type flattened_bounds: List[tuple]
        :return: Unflattened bounds matching original structure
        :rtype: List[Union[tuple, List[tuple]]]
        
        Example:
            For CalParams with modes ['direct', 'linear', 'multiplicative']:
            Input:  [(0, 1), (0.5, 1.5), (-1, 1), (2, 3)]
            Output: [(0, 1), [(0.5, 1.5), (-1, 1)], (2, 3)]
        """
        unflattened = []
        idx = 0
        
        for cp in self:
            if not cp.active:
                continue
                
            if cp.mode == "linear":
                # Linear mode has two bounds (slope, intercept)
                if idx + 1 >= len(flattened_bounds):
                    raise ValueError(
                        f"Not enough bounds for linear parameter {cp.section}.{cp.attribute}. "
                        f"Expected 2 bounds, but only {len(flattened_bounds) - idx} remaining."
                    )
                unflattened.append([flattened_bounds[idx], flattened_bounds[idx + 1]])
                idx += 2
            else:
                # Regular parameter has one bound
                if idx >= len(flattened_bounds):
                    raise ValueError(
                        f"Not enough bounds for parameter {cp.section}.{cp.attribute}. "
                        f"Expected 1 bound, but no bounds remaining."
                    )
                unflattened.append(flattened_bounds[idx])
                idx += 1
        
        if idx != len(flattened_bounds):
            raise ValueError(
                f"Bounds length mismatch. Expected {idx} bounds, got {len(flattened_bounds)}."
            )
            
        return unflattened

    def flatten_values(self, values: List[Union[float, tuple, List[float]]]) -> List[float]:
        """
        Flatten a list of optimization values that may contain single values or tuples.
        
        For linear mode parameters, each parameter may have multiple values
        (slope, intercept). This function flattens them into a single list.
        
        :param values: List of values, where each element is either a float
                      or a tuple/list of floats for linear mode
        :type values: List[Union[float, tuple, List[float]]]
        :return: Flattened list of values
        :rtype: List[float]
        
        Example:
            Input:  [0.5, (1.2, -0.3), 2.1]
            Output: [0.5, 1.2, -0.3, 2.1]
        """
        flattened = []
        for value in values:
            if isinstance(value, (tuple, list)):
                # Linear mode values - extend with all sub-values
                flattened.extend(value)
            elif isinstance(value, (int, float)):
                # Regular value - append as single value
                flattened.append(float(value))
            else:
                raise TypeError(f"Values must be number, tuple, or list, got {type(value)}")
        return flattened

    def unflatten_values(self, flattened_values: List[float]) -> List[Union[float, tuple]]:
        """
        Unflatten a list of optimization values back to the original structure.
        
        This is the inverse of flatten_values, reconstructing the original
        values structure based on the parameter modes.
        
        :param flattened_values: Flattened list of values
        :type flattened_values: List[float]
        :return: Unflattened values matching original structure
        :rtype: List[Union[float, tuple]]
        
        Example:
            For CalParams with modes ['direct', 'linear', 'multiplicative']:
            Input:  [0.5, 1.2, -0.3, 2.1]
            Output: [0.5, (1.2, -0.3), 2.1]
        """
        unflattened = []
        idx = 0
        
        for cp in self:
            if not cp.active:
                continue
                
            if cp.mode == "linear":
                # Linear mode has two values (slope, intercept)
                if idx + 1 >= len(flattened_values):
                    raise ValueError(
                        f"Not enough values for linear parameter {cp.section}.{cp.attribute}. "
                        f"Expected 2 values, but only {len(flattened_values) - idx} remaining."
                    )
                unflattened.append((flattened_values[idx], flattened_values[idx + 1]))
                idx += 2
            else:
                # Regular parameter has one value
                if idx >= len(flattened_values):
                    raise ValueError(
                        f"Not enough values for parameter {cp.section}.{cp.attribute}. "
                        f"Expected 1 value, but no values remaining."
                    )
                unflattened.append(flattened_values[idx])
                idx += 1
        
        if idx != len(flattened_values):
            raise ValueError(
                f"Values length mismatch. Expected {idx} values, got {len(flattened_values)}."
            )
            
        return unflattened

    def filter_by_section(self, section: str) -> "CalParams":
        """Filter parameters by section."""
        return CalParams([cp for cp in self if cp.section == section])

    def distribute(self, model: swmmio.Model) -> "CalParams":
        """Distribute all parameters in the list."""
        distributed = []
        
        for cp in self:
            if cp.distributed:
                distributed.extend(cp.distribute(model))
            else:
                distributed.append(cp)
        return CalParams(distributed)

    def get_initial_values(self, model: swmmio.Model) -> "CalParams":
        """Extract initial values from the model."""
        valid_cps = []

        for cp in self:
            if not cp.distributed:
                 cp.initial_value = getattr(model.inp, cp.section)[cp.attribute].values
            else:
                if len(cp.key) > 1:
                    # Multi-indexed parameter
                    index = cp.make_multi_index(model)
                    swmm_df = getattr(model.inp, cp.section).copy()
                    swmm_df.set_index(index, inplace=True)
                    cp.initial_value = swmm_df.loc[cp.element, cp.attribute]
                else:
                    cp.initial_value = getattr(model.inp, cp.section).loc[cp.element, cp.attribute]
                    
                    if np.array(cp.initial_value).size > 1:
                        cp.initial_value = cp.initial_value[cp.row_index]


            # Some optimization algorithms require non-zero initial guesses
            # These are dependant on the mode of the parameter
            # For example, a multiplicative calibration parameter with an initial value of 1.01
            # will have an initial guess of 0.01 * initial_value (1% increase); whereas an additive
            # parameter with an initial value of 0.01 will have an initial guess of 0.01 + initial_value

            # Set initial guess
            initial_cal_values = {
                "multiplicative": 1.01,
                "additive": 0.01,
                "direct": cp.initial_value + 0.01,
                "linear": (0.01, 1.05)
            }

            # Set initial guess based on mode
            cp.x0 = initial_cal_values[cp.mode] #cp.set_model_value(initial_value=cp.initial_value, cal_value=initial_cal_values[cp.mode])

            # Only include parameters with valid initial values
            if not np.isnan(np.mean(cp.initial_value)):
                valid_cps.append(cp)

        # Set numeric indices for mapping
        for ii, cp in enumerate(valid_cps):
            cp.ii = ii
            
        return CalParams(valid_cps)

    def rel_to_abs_bounds(self, upper: Optional[float] = None, lower: Optional[float] = None) -> "CalParams":
        """Convert multiplicative bounds to absolute for all non-multiplicative parameters."""
        for cp in self:
            if cp.upper_abs is None or cp.lower_abs is None:
                cp.rel_to_abs_bounds(upper=upper, lower=lower)
        return self

    def set_values(self, values: List[float]|List[tuple[float,float]]):
        """Set optimization values for each parameter."""
        #if len(values) != len(self):
        #    raise ValueError("Length of values must match length of calibration parameters.")

        values = self.unflatten_values(values)

        for cp, val in zip(self, values):
            if cp.active:
                cp.opt_value = val
                
                if cp.mode == "multiplicative":
                    cp.model_value = cp.initial_value * (1 + val)
                elif cp.mode == "additive":
                    cp.model_value = cp.initial_value + val
                elif cp.mode == "direct":
                    cp.model_value = val
                elif cp.mode == "linear":
                    if len(val) != 2:
                        raise ValueError("Linear mode requires a tuple of (slope, intercept).")
                    cp.model_value = val[1] * cp.initial_value + val[0]

    def make_simulation_preconfig(self, model: swmmio.Model) -> SimulationPreConfig:
        """Create SimulationPreConfig with parameter updates."""
        spc = SimulationPreConfig()

        for cp in self:

            # val might be a single value or an array of values (if the changes are relative)
            val = cp.model_value
            val = cp.truncate(val)

            # ensure val is a numpy array for consistent processing
            if not isinstance(val, np.ndarray):
                val = np.array(val)

        
            if not cp.distributed:
                if cp.mode not in ["multiplicative", "additive", "linear"]:
                    raise NotImplementedError("Non-distributed calibration params must be relative")

                section_df = getattr(model.inp, cp.section)
                element_ids = section_df.index.to_list()
                if val.size == 1:
                    val = [val] * len(element_ids)

                for element_id, new_val in zip(element_ids, val):
                    spc.add_update_by_token(
                        section=cp.section,
                        obj_id=element_id,
                        index=cp.col_index,
                        new_val=new_val,
                        row_num=cp.row_index
                    )
            else:
                spc.add_update_by_token(
                    section=cp.section,
                    obj_id=cp.element,
                    index=cp.col_index,
                    new_val=val,
                    row_num=cp.row_index
                )

        return spc

    def get_calibration_order(self, model: swmmio.Model) -> "CalParams":
        """Sort parameters by calibration order based on network topology."""
        node_level = {node: len(get_upstream_nodes(model.network, node)) 
                     for node in model.network.nodes}
        
        for ii, _ in enumerate(self):
            if self[ii].parent_section == "subcatchments":
                node = model.inp.subcatchments.loc[self[ii].element, 'Outlet']
                self[ii].level = node_level[node]
                self[ii].node = node
            elif self[ii].parent_section in ["conduit", "conduits", "link", "links"]:
                node = model.inp.conduits.loc[self[ii].element, 'InletNode']
                self[ii].level = node_level[node]
                self[ii].node = node
            elif self[ii].parent_section in ["nodes", "node"]:
                self[ii].level = node_level[self[ii].element]
                self[ii].node = self[ii].element
            else:
                self[ii].level = 0
                self[ii].node = "all"

        # Add small increments to avoid level collisions
        nodes = [cp.node for cp in self if cp.node != "None"]
        unique_nodes = np.unique(nodes).tolist()
        level_increment = 1 / (len(unique_nodes) + 1)
        level_adjustments = {node: ii * level_increment for ii, node in enumerate(unique_nodes)}

        for ii, _ in enumerate(self):
            if self[ii].node in level_adjustments:
                self[ii].level += level_adjustments[self[ii].node]

        return self

    def to_df(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return _cons_to_df(self)

    def calculate_bounds(self):
        """Process bounds to ensure they are in the correct format."""
        for cp in self:
            cp.calculate_bounds()
        return self

    def preprocess(self, opt):
        """Preprocess parameters for calibration."""

        model = Model(str(opt.model_file))
        # Distribute parameters if needed
        self = self.distribute(model=model)

        # Get initial values from model
        self = self.get_initial_values(model=model)

        self = self.calculate_bounds()

        if ~np.any([cp.distributed for cp in self]) and opt.hierarchical:
            raise ValueError("No distributed parameters found, hierarchical calibration cannot be applied.")
        # Set calibration order based on network topology
        if opt.hierarchical:
            self = self.get_calibration_order(model=model)
        return self
    
    @classmethod
    def from_file(cls, filename: Union[Path, str]) -> "CalParams":
        """
        Create CalParams from CSV file.
        
        :param filename: Path to CSV file containing calibration parameters
        :type filename: Path or str
        :return: CalParams object
        :rtype: CalParams
        """
        if isinstance(filename, str):
            filename = Path(filename)
            
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} does not exist.")
            
        df = pd.read_csv(filename, index_col=0)
        df = df.replace({np.nan: None})

        cps = []
        for _, row in df.iterrows():
            # Handle key field which might be a string representation of a list
            key = row['key']
            if isinstance(key, str) and key.startswith('['):
                try:
                    key = eval(key)
                except:
                    key = [key]
            elif isinstance(key, str):
                key = [key]
                
            cp = CalParam(
                section=row['section'],
                attribute=row['attribute'],
                element=row.get('element', ''),
                key=key,
                lower=row['lower'],
                upper=row['upper'],
                lower_limit=row['lower_limit'],
                upper_limit=row['upper_limit'],
                distributed=row['distributed'],
                mode=row.get('mode', 'direct'),
                relative_bounds=row.get('relative_bounds', True),
                col_index=row.get('col_index'),
                initial_value=row.get('initial_value', np.nan),
                opt_value=row.get('opt_value', np.nan),
                model_value=row.get('model_value', np.nan),
                row_attribute=row.get('row_attribute'),
                row_index=row.get('row_index', 0),
                level=row.get('level', 0),
                ii=row.get('ii', 0),
                active=row.get('active', True)
            )
            cps.append(cp)
            
        return cls(cps)


def _cons_to_df(cons: CalParams) -> pd.DataFrame:
    """Convert CalParams to DataFrame."""
    data = {
        'section': [cp.section for cp in cons],
        'attribute': [cp.attribute for cp in cons],
        'element': [cp.element for cp in cons],
        'key': [cp.key for cp in cons],
        'ii': [cp.ii for cp in cons],
        'col_index': [cp.col_index for cp in cons],
        'lower_rel': [cp.lower_rel for cp in cons],
        'upper_rel': [cp.upper_rel for cp in cons],
        'lower_abs': [cp.lower_abs for cp in cons],
        'upper_abs': [cp.upper_abs for cp in cons],
        'lower_limit': [cp.lower_limit_abs for cp in cons],
        'upper_limit': [cp.upper_limit_abs for cp in cons],
        'distributed': [cp.distributed for cp in cons],
        'mode': [cp.mode for cp in cons],
        'relative_bounds': [cp.relative_bounds for cp in cons],
        'initial_value': [cp.initial_value for cp in cons],
        'x0': [cp.x0 for cp in cons],
        'row_index': [cp.row_index for cp in cons],
        'row_attribute': [cp.row_attribute for cp in cons],
        'parent_section': [cp.parent_section for cp in cons],
        'tag': [cp.tag for cp in cons],
        'level': [getattr(cp, 'level', None) for cp in cons],
        'node': [getattr(cp, 'node', None) for cp in cons],
    }
    return pd.DataFrame(data)


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
    cps_distributed = cps_distributed.relative_to_absolute_search_bounds()

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
                if cp.mode not in ["multiplicative", "additive", "linear"]:
                    raise NotImplementedError("Non-distributed calibration params must be set to 'multiplicative'")
                
                # the calibrated value 'val' is the multiplicative change - so we need to calculate the new model values multiplicative to the initial values
                val_map = getattr(Model(model_file).inp, cp.section).loc[:,cp.attribute].to_dict()
                for element_id, model_val in val_map.items():
                    new_val = model_val * (1 + val)
                    new_val = cp.truncate(new_val)
                    spc.add_update_by_token(section=cp.section, obj_id=element_id, col_index=cp.col_index, new_val=new_val)
        
            # if it's a distributed calibration parameter
        else:
            # if the calibrated value 'val' is the multiplicative change - so we need to calculate the new model values multiplicative to the initial values
            
            new_val = calculate_new_val(cp, val)

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
                lower_rel=row.lower_rel,
                upper_rel=row.upper_rel,
                lower_abs=row.lower_abs,
                upper_abs=row.upper_abs,
                upper_limit=row.upper_limit,
                lower_limit=row.lower_limit,
                mode=row.mode,
                distributed=row.distributed,
                initial_value = row.initial_value)
        cps.append(cp)
    cps = CalParams(cps)

    model_file = str(run_dir.parent.parent / f"{run_dir.parent.parent.stem}_cal.inp")

    
    spc=create_simpreconfig(cps=cps, vals=df.cal_val, model_file=model_file)


    return spc