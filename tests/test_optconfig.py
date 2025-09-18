import pytest
from optswmm.utils.optconfig import OptConfig

def test_optconfig_initialization():
    config = OptConfig(model_file='path/to/model.inp', output_dir='path/to/output')
    assert config.model_file == 'path/to/model.inp'
    assert config.output_dir == 'path/to/output'

def test_optconfig_validation():
    config = OptConfig(model_file='path/to/model.inp', output_dir='path/to/output')
    assert config.validate() is True

def test_optconfig_invalid_model_file():
    config = OptConfig(model_file='', output_dir='path/to/output')
    with pytest.raises(ValueError, match="Model file must be specified"):
        config.validate()

def test_optconfig_invalid_output_dir():
    config = OptConfig(model_file='path/to/model.inp', output_dir='')
    with pytest.raises(ValueError, match="Output directory must be specified"):
        config.validate()