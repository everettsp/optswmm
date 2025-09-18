import pytest
from optswmm.utils.calparams import CalParam, CalParams

def test_calparam_initialization():
    param = CalParam(
        section='subcatchments',
        attribute='Width',
        lower_rel=0.1,
        upper_rel=1.0,
        lower_abs=0.5,
        upper_abs=2.0,
        distributed=True,
        mode='additive',
        initial_value=1.0
    )
    
    assert param.section == 'subcatchments'
    assert param.attribute == 'Width'
    assert param.lower_rel == 0.1
    assert param.upper_rel == 1.0
    assert param.lower_abs == 0.5
    assert param.upper_abs == 2.0
    assert param.distributed is True
    assert param.mode == 'additive'
    assert param.initial_value == 1.0

def test_calparam_truncate():
    param = CalParam(
        section='subcatchments',
        attribute='Width',
        lower_limit_abs=0.0,
        upper_limit_abs=2.0,
        initial_value=1.0
    )
    
    assert param.truncate(-1.0) == 0.0
    assert param.truncate(3.0) == 2.0
    assert param.truncate(1.0) == 1.0

def test_calparams_get_bounds():
    params = CalParams([
        CalParam(section='subcatchments', attribute='Width', lower_abs=0.5, upper_abs=2.0),
        CalParam(section='subcatchments', attribute='Area', lower_abs=1.0, upper_abs=5.0)
    ])
    
    bounds = params.get_bounds()
    assert bounds == [(0.5, 2.0), (1.0, 5.0)]

def test_calparams_flatten_bounds():
    params = CalParams([
        CalParam(section='subcatchments', attribute='Width', lower_abs=0.5, upper_abs=2.0),
        CalParam(section='subcatchments', attribute='Area', lower_abs=1.0, upper_abs=5.0)
    ])
    
    flattened = params.flatten_bounds([(0.5, 2.0), [(1.0, 5.0)]])
    assert flattened == [(0.5, 2.0), (1.0, 5.0)]

def test_calparams_unflatten_bounds():
    params = CalParams([
        CalParam(section='subcatchments', attribute='Width', lower_abs=0.5, upper_abs=2.0),
        CalParam(section='subcatchments', attribute='Area', lower_abs=1.0, upper_abs=5.0)
    ])
    
    unflattened = params.unflatten_bounds([(0.5, 2.0), (1.0, 5.0)])
    assert unflattened == [(0.5, 2.0), (1.0, 5.0)]