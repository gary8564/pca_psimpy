import pytest
import numpy as np
from beartype.roar import BeartypeCallHintParamViolation
from psimpy.utility.util_funcs import scale_samples


@pytest.mark.parametrize(
    'samples, bounds',
    [
     ([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]], [[2, 4], [20, 40]]),
     ([[0.3], [0.1]], [[0, 10]])
     ]
)
def test_scale_samples_TypeError(samples, bounds):
    with pytest.raises(BeartypeCallHintParamViolation):
        scale_samples(samples, bounds)

@pytest.mark.parametrize(
    'samples, bounds',
    [
     (np.array([0.1, 0.5, 0.9]), np.array([[2, 4]])),
     (np.array([[0.1], [0.5], [0.9]]), np.array([2, 4])),
     (np.array([[0.1], [0.5], [0.9]]), np.array([[2, 4], [20, 40]])),
     (np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]]),
      np.array([[2, 4], [20, 10]])
      )
     ]
)
def test_scale_samples_ValueError(samples, bounds):
    with pytest.raises(ValueError):
        scale_samples(samples, bounds)

@pytest.mark.parametrize(
    'samples, bounds, results',
    [
     (np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]]), 
      np.array([[2, 4], [20, 40]]),
      np.array([[2.2, 24], [2.8, 30], [3.4, 36]])
      ),
     (np.array([[0.3], [0.1]]),
      np.array([[0, 10]]),
      np.array([[3], [1]])
      )
     ]
)
def test_scale_samples(samples, bounds, results):
    scaled_samples = scale_samples(samples, bounds)
    assert np.all(scaled_samples == np.array(results))
    
