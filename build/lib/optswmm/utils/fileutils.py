import os
from datetime import datetime
from pathlib import Path

def initialize_run(run_loc: Path, name: str):
    """
    Creates a timestamped run directory to track calibration results.

    :param run_loc: Path to the directory where the run directory will be created.
    :type run_loc: Path
    :param name: Name of the calibration routine.
    :type name: str
    :returns: Path to the created run directory.
    :rtype: Path
    """
    now = datetime.now()
    current_time = now.strftime("%d-%m-%y-%H%M%S")
    run_dir = "run-{}_{}".format(name, current_time)
    x = Path(os.path.join(run_loc, run_dir))
    os.mkdir(x)
    
    return x