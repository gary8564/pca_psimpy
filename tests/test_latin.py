import pytest
import numpy as np
from psimpy.sampler import LHS
from beartype.roar import BeartypeCallHintParamViolation

@pytest.mark.parametrize(
    "ndim, bounds, seed, criterion, iteration",
    [
        (2.0, np.array([[1,2],[10,20]]), 1, 'random', 100),
        (2, [[1,2],[10,20]], 1, 'random', 100),
        (2, np.array([[1,2],[10,20]]), 1.0, 'random', 100),
        (2, np.array([[1,2],[10,20]]), 1, None, 100),
        (2, np.array([[1,2],[10,20]]), 1, 'center', 100.0)
    ]
)
def test_init_TypeError(ndim, bounds, seed, criterion, iteration):
    with pytest.raises(BeartypeCallHintParamViolation):
        LHS(ndim, bounds, seed, criterion, iteration)

   
@pytest.mark.parametrize(
    "criterion",
    ['max','Random','maxmin']
)
def test_init_NotImplementedError(criterion):
    with pytest.raises(NotImplementedError):
        LHS(3, criterion=criterion)
        
 
@pytest.mark.parametrize(
    "ndim, bounds, seed, criterion, iteration, nsamples",
    [
        (2, None, None, 'random', None, 10), 
        (2, np.array([[2, 4], [20, 40]]), 123, 'random', None, 10), 
        (2, None, 123, 'random', None, 10), 
        (3, np.array([(8, 9), (5, 10), (2, 10)]), 123, 'center', None, 10), 
        (3, None, 45, 'center', None, 10), 
        (4, None, 220, 'maximin', None, 5), 
        (2, np.array([[1, 20], [0, 2]]), None, 'maximin', 200, 20),
        (5, None, 100, 'maximin', 100, 10)
    ]
)       
def test_sample(ndim, bounds, seed, criterion, iteration, nsamples):
    a = LHS(ndim, bounds, seed, criterion, iteration)
    samples = a.sample(nsamples)
    assert a.nsamples == nsamples
    if iteration is not None:
        assert a.iteration == iteration
    elif criterion == 'maximin':
        assert a.iteration == 100
    for j in range(ndim):
        counts, _ = np.histogram(samples[:,j], bins=nsamples, 
            range=(0, 1) if bounds is None else bounds[j, :])
        res = [(item==1) for item in counts]
        assert all(res)


@pytest.mark.parametrize(
    "seed1, seed2, expected",
    [
        (123, 123, True), 
        (123, 120, False), 
        (111, 111, True), 
        (111, 2, False)
    ]
)       
def test_sample_with_seed(seed1, seed2, expected):
    a = LHS(3, seed=seed1)
    a_samples = a.sample(10)
    b = LHS(3, seed=seed2)
    b_samples = b.sample(10)
    assert np.array_equal(a_samples, b_samples) == expected