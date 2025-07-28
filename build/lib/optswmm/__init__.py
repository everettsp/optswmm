"""
OptSWMM - SWMM optimization and calibration utilities
"""

__version__ = "0.1.0"
__author__ = "Everett"

from .utils.calparams import CalParam, CalParams

__all__ = ["CalParam", "CalParams"]

"""
optswmm
=======

Simple package for optimizing SWMM parameters.

This package provides tools for calibrating SWMM models through various
optimization techniques.
"""

# Import key modules and functions to make them available at package level
try:
    from optswmm.utils.calibutils import calibrate, get_scaler, normalise, denormalise
    from optswmm.utils.calparams import CalParam, CalParams, get_cal_params
    from optswmm.utils.optconfig import OptConfig
    from optswmm.utils.swmmutils import (
        get_node_timeseries,
        get_link_timeseries, 
        get_subcatchment_timeseries,
        get_model_datetimes,
        set_model_datetimes,
        run_swmm
    )
    from optswmm.utils.runutils import summarize_runs, initialize_run
except ImportError:
    # Fallback to relative imports (for development environment)
    from .utils.calibutils import calibrate, get_scaler, normalise, denormalise
    from .utils.calparams import CalParam, CalParams, get_cal_params
    from .utils.optconfig import OptConfig
    from .utils.swmmutils import (
        get_node_timeseries,
        get_link_timeseries, 
        get_subcatchment_timeseries,
        get_model_datetimes,
        set_model_datetimes,
        run_swmm
    )
    from .utils.runutils import summarize_runs, initialize_run

# Define what gets imported with "from optswmm import *"
__all__ += [
    'calibrate',
    'OptConfig',
    'get_cal_params',
    'get_node_timeseries',
    'get_link_timeseries',
    'get_subcatchment_timeseries',
    'get_model_datetimes',
    'set_model_datetimes',
    'run_swmm',
    'summarize_runs',
    'initialize_run',
]