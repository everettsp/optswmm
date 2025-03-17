



from swmmio import Model
from pathlib import Path
from utils.optconfig import OptConfig



data_dir = Path(r"C:\Users\everett\Documents\GitHub\camus_to")
#mdl = Model(str(data_dir/"test.inp"))
from defs import load_yaml


opt_config = OptConfig(Path() / 'defs' / 'default_OptConfig.yaml')

print(opt_config)
#print(mdl.inp.subcatchments)

from utils.calibutils import calibrate

calibrate(opt_config)