import pytest
import numpy as np
from psimpy.sampler import Saltelli
from psimpy.sampler import LHS
from psimpy.emulator import ScalarGaSP
from psimpy.sensitivity import SobolAnalyze

def f(x1,x2,x3):
    return np.sin(x1) + 7*np.sin(x2)**2 + 0.1*x3**4*np.sin(x1)


@pytest.mark.parametrize(
    "calc_second_order, skip_values, nbase, seed, mode, max_workers",
    [
        (False, None, 1024, None, None, None),
        (True, None, 1024, 1, 'parallel', 2),
        (True, 2**16, 2048, 10, 'serial', None),
        (False, 2**16, 2048, 8, 'serial', None)
    ]
)       
def test_SobolAnalyze(calc_second_order, skip_values, nbase, seed, mode,
    max_workers):
    
    ndim = 3
    bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    saltelli_sampler = Saltelli(ndim, bounds, calc_second_order, skip_values)
    saltelli_samples = saltelli_sampler.sample(nbase)
    Y = f(saltelli_samples[:,0], saltelli_samples[:,1], saltelli_samples[:, 2])

    sobol_analyzer = SobolAnalyze(ndim, Y, calc_second_order, seed=seed)
    S_res = sobol_analyzer.run(mode, max_workers)

    assert 'S1' in S_res.keys()
    assert 'ST' in S_res.keys()
    if calc_second_order:
        assert 'S2' in S_res.keys()
    else:
        assert 'S2' not in S_res.keys()

    print('\n', 'estimated S1 (mean, std, conf_level): \n', S_res['S1'])
    print('analytical S1: \n', [0.314, 0.442, 0])


@pytest.mark.parametrize(
    "calc_second_order, skip_values, nbase, seed, mode, max_workers",
    [
        (False, None, 1024, None, None, None),
        (True, None, 1024, 1, 'parallel', 4),
        (False, None, 1024, 2, 'parallel', 4),
        (True, 2**16, 512, 10, 'serial', None),
        (False, 2**16, 512, 8, 'serial', None)
    ]
)       
def test_SobolAnalyze_with_emulation(calc_second_order, skip_values, nbase,
    seed, mode, max_workers):

    ndim = 3
    bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])

    lhs_sampler = LHS(ndim, bounds, seed, criterion='maximin', iteration=200)
    design = lhs_sampler.sample(nsamples=100)
    response = f(design[:,0], design[:,1], design[:,2])
    
    scalar_gasp = ScalarGaSP(ndim)
    print('\n')
    scalar_gasp.train(design, response)
    validation = scalar_gasp.loo_validate()
    relative_diff = (response - validation[:,0]) / (np.max(response)- 
        np.min(response))
    print('\n', 'emulator max relative_diff: ',
        f'{np.max(np.abs(relative_diff))}')
    print('emulator mean relative_diff: ',
        f'{np.mean(np.abs(relative_diff))}')
    
    saltelli_sampler = Saltelli(ndim, bounds, calc_second_order, skip_values)
    saltelli_samples = saltelli_sampler.sample(nbase)
    
    Y = scalar_gasp.sample(saltelli_samples, 50)

    sobol_analyzer = SobolAnalyze(ndim, Y, calc_second_order, seed=seed)
    S_res = sobol_analyzer.run(mode, max_workers)

    assert 'S1' in S_res.keys()
    assert 'ST' in S_res.keys()
    if calc_second_order:
        assert 'S2' in S_res.keys()
    else:
        assert 'S2' not in S_res.keys()

    print('\n', 'estimated S1 (mean, std, conf_level): \n', S_res['S1'])
    print('analytical S1: \n', [0.314, 0.442, 0])