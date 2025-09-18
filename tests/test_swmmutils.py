import pytest
from optswmm.utils.swmmutils import get_node_timeseries, get_link_timeseries, get_subcatchment_timeseries, run_swmm
from pathlib import Path

@pytest.fixture
def model_path():
    return Path(__file__).parent / 'fixtures/sample_models/basic_model.inp'

def test_get_node_timeseries(model_path):
    timeseries = get_node_timeseries(model_path, 'node_id')
    assert timeseries is not None
    assert len(timeseries) > 0

def test_get_link_timeseries(model_path):
    timeseries = get_link_timeseries(model_path, 'link_id')
    assert timeseries is not None
    assert len(timeseries) > 0

def test_get_subcatchment_timeseries(model_path):
    timeseries = get_subcatchment_timeseries(model_path, 'subcatchment_id')
    assert timeseries is not None
    assert len(timeseries) > 0

def test_run_swmm(model_path):
    try:
        run_swmm(model_path)
    except Exception as e:
        pytest.fail(f"run_swmm raised an exception: {e}")