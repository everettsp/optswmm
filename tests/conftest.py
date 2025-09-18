# filepath: tests/conftest.py
import pytest
import os

@pytest.fixture(scope='session')
def sample_model_path():
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_models')

@pytest.fixture(scope='session')
def sample_data_path():
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_data')

@pytest.fixture(scope='session')
def config_path():
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'configs')