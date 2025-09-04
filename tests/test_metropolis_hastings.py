import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, uniform
from psimpy.sampler import MetropolisHastings

@pytest.mark.parametrize(
    "ndim, init_state, f_sample, target, ln_target, bounds, f_density, symmetric",
    [
        (1, np.array([1,1]), norm.rvs, norm.pdf, None, None, None, True), 
        (1, np.array([0.5]), norm.rvs, uniform.pdf, None, np.array([0,1]),
        None, True),
        (1, np.array([0.5]), norm.rvs, uniform.pdf, None, np.array([[0],[1]]),
        None, True),
        (1, np.array([1.5]), norm.rvs, uniform.pdf, None, np.array([[0,1]]),
        None, True),
        (2, np.array([-1,1]), multivariate_normal.rvs, multivariate_normal.pdf,
        None, None, None, False)
    ]
)
def test_init_ValueError(ndim, init_state, f_sample, target, ln_target, bounds,
    f_density, symmetric):
    with pytest.raises(ValueError):
        _ = MetropolisHastings(ndim=ndim, init_state=init_state,
            f_sample=f_sample, target=target, ln_target=ln_target,
            bounds=bounds, f_density=f_density, symmetric=symmetric)


def test_sample_ValueError():
    mh_sampler = MetropolisHastings(ndim=2, init_state=np.array([-1,1]),
        f_sample=multivariate_normal.rvs, target=None, ln_target=None)
    with pytest.raises(ValueError):
        mh_sampler.sample(nsamples=10000)


def test_sample_uniform_target():
    ndim = 1
    init_state = np.array([0.5])
    f_sample = norm.rvs
    target = uniform.pdf
    bounds = np.array([[0,1]])
    nburn = 100
    nthin = 10
    seed = 1
    kwgs_f_sample = {'scale':1, 'random_state': np.random.default_rng(seed)}

    mh_sampler = MetropolisHastings(ndim=ndim, init_state=init_state,
        f_sample=f_sample, target=target, bounds=bounds, nburn=nburn,
        nthin=nthin, seed=seed, kwgs_f_sample=kwgs_f_sample)
    nsamples = 10000
    mh_samples, mh_accept = mh_sampler.sample(nsamples)
    assert mh_samples.shape == (nsamples, ndim)
    assert mh_accept.shape == (nsamples,)
    print('mean: ', np.mean(mh_samples))
    print('expected mean: ', 0.5)
    print('std: ', np.std(mh_samples))
    print('expected std: ', np.sqrt(1/12))
    print('acceptance ratio: ', np.mean(mh_accept))

    hist, _ = np.histogram(mh_samples, bins=10, range=(0,1))
    print('hist: ', hist/mh_samples.size)
    print('expected hist: ', [0.1]*10)


def test_sample_multivariate_norm_target():
    ndim = 2
    init_state = np.array([1,1])
    f_sample = multivariate_normal.rvs
    ln_target = multivariate_normal.logpdf
    nburn = 1000
    nthin = 5
    seed = 88
    kwgs_f_sample = {'random_state': np.random.default_rng(seed)}
    kwgs_target = {'mean':[0, 2], 'cov':[[1, 0.5],[-0.5, 5]]}

    mh_sampler = MetropolisHastings(ndim=ndim, init_state=init_state,
        f_sample=f_sample, ln_target=ln_target, nburn=nburn, nthin=nthin,
        seed=seed, kwgs_f_sample=kwgs_f_sample, kwgs_target=kwgs_target)
    nsamples = 10000
    mh_samples, mh_accept = mh_sampler.sample(nsamples)
    assert mh_samples.shape == (nsamples, ndim)
    assert mh_accept.shape == (nsamples,)
    print('mean: ', np.mean(mh_samples, axis=0))
    print('expected mean: ', [0, 2])
    print('std: ', np.std(mh_samples, axis=0))
    print('expected std: ', np.sqrt([1, 5]))
    print('acceptance ratio: ', np.mean(mh_accept))

    # plot
    x, y = np.mgrid[-4:4:0.01, -6:10:0.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(kwgs_target['mean'], kwgs_target['cov'])

    fig, axes = plt.subplots(1,2,figsize=(8,4))
    axes[0].contourf(x,y, rv.pdf(pos), levels=20)
    axes[1].scatter(mh_samples[:,0], mh_samples[:,1], c='r', marker='o',
        alpha=0.05)
    for i in range(2):
        axes[i].set_xlim(-4,4)
        axes[i].set_ylim(-6,10)
    axes[0].set_title('True target')
    axes[1].set_title('MH samples')

    dir_test = os.path.abspath(os.path.join(__file__, '../'))
    os.chdir(dir_test)
    if not os.path.exists('temp_metropolis_hastings'):
        os.mkdir('temp_metropolis_hastings')
    dir_out = os.path.join(dir_test, 'temp_metropolis_hastings')
    plt.savefig(os.path.join(dir_out,'2d_norm_target.png'), bbox_inches='tight')


