import pytest
import numpy as np
from SALib.sample.saltelli import sample
from psimpy.sampler import Saltelli
from beartype.roar import BeartypeCallHintParamViolation


@pytest.mark.parametrize(
    "ndim, bounds, calc_second_order, skip_values",
    [
        (3.0, None, False, None),
        (3, [[1,2],[10,20],[100,200]], False, None),
        (3, None, 1, None),
        (3, None, 1, False)
    ]
)
def test_init_TypeError(ndim, bounds, calc_second_order, skip_values):
    with pytest.raises(BeartypeCallHintParamViolation):
        _ = Saltelli(ndim, bounds, calc_second_order, skip_values)


@pytest.mark.parametrize(
    "ndim, bounds, calc_second_order, skip_values, nbase, problem",
    [
        (3, None, True, None, 2**5, {'names':['x1', 'x2', 'x3'], 
        'num_vars':3, 'bounds':np.array([[0, 1], [0, 1], [0, 1]])}
        ), 
        (2, np.array([[2, 4], [20, 40]]), False, 16, 2**4, {'names':['x1', 'x2'], 
        'num_vars':2, 'bounds':np.array([[2, 4], [20, 40]])}
        ), 
        (3, None, False, None, 2**6,  {'names':['x1', 'x2', 'x3'], 
        'num_vars':3, 'bounds':np.array([[0, 1], [0, 1], [0, 1]])}
        ), 
        (2, np.array([[2, 4], [20, 40]]), True, 2**10, 2**8, {'names':['x1', 'x2'], 
        'num_vars':2, 'bounds':np.array([[2, 4], [20, 40]])}
        ) 
    ]
)       
def test_sample(ndim, bounds, calc_second_order, skip_values, nbase, problem):
    saltelli_sampler = Saltelli(ndim, bounds, calc_second_order, skip_values)
    samples = saltelli_sampler.sample(nbase)
    expected = sample(problem, N=nbase, calc_second_order=calc_second_order, 
        skip_values=skip_values)
    coeff = 2 if calc_second_order else 1
    assert samples.shape == (nbase * (coeff*ndim + 2), ndim)
    assert np.array_equal(samples, expected)