import os
import pytest
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, uniform
from psimpy.inference import GridEstimation
from psimpy.inference import MetropolisHastingsEstimation
from psimpy.sampler import MetropolisHastings

dir_test = os.path.abspath(os.path.join(__file__, '../'))

@pytest.mark.parametrize(
    "ndim, bounds, prior, likelihood, ln_prior, ln_likelihood, ln_pxl, nbins",
    [
        (1, np.array([0,1]), uniform.pdf, norm.pdf, None, None, None, 10), 
        (1, np.array([[0],[1]]), uniform.pdf, norm.pdf, None, None, None, 10),
        (1, np.array([[0,1]]), None, None, None, None, None, 10),
        (1, np.array([[0,1]]), None, None, None, norm.logpdf, None, 10),
        (1, np.array([[0,1]]), uniform.pdf, None, None, None, None, 10),        
        (1, np.array([[0,1]]), None, None, None, None, norm.logpdf, [10,10]),
        (1, None, None, None, None, None, norm.logpdf, [10]),
    ]
)
def test_GridEstimation_ValueError(ndim, bounds, prior, likelihood, ln_prior,
    ln_likelihood, ln_pxl, nbins):
    with pytest.raises(ValueError):
        grid_estimator = GridEstimation(ndim, bounds, prior, likelihood, 
            ln_prior, ln_likelihood, ln_pxl)
        grid_estimator.run(nbins)


def test_GridEstimation_1d():
    ndim = 1
    bounds = np.array([[-1,1]])
    prior = uniform.pdf
    kwgs_prior = {'loc': -1, 'scale': 2}
    likelihood = norm.pdf
    nbins = 100
    grid_estimator = GridEstimation(ndim, bounds, prior, likelihood,
        kwgs_prior=kwgs_prior)
    posterior, x_ndim = grid_estimator.run(nbins)
    assert posterior.shape == (nbins,)
    assert len(x_ndim) == ndim
    assert x_ndim[0].shape == (nbins,)
    assert np.abs(np.sum(posterior)*(bounds[0,1]-bounds[0,0])/nbins - 1) < 1e-10

    # plot
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.plot(x_ndim[0], posterior, 'b-')
    ax.set_xlim(bounds[0,0] - 0.5, bounds[0,1] + 0.5)
    ax.set_ylim(0, np.max(posterior)*1.1)
    ax.set_title('Grid estimation, prior U[-1,1], likelihood N(0,1)')
    ax.set_xlabel('x')
    ax.set_ylabel('pdf')

    os.chdir(dir_test)
    if not os.path.exists('temp_bayes_inference'):
        os.mkdir('temp_bayes_inference')
    dir_out = os.path.join(dir_test, 'temp_bayes_inference')
    plt.savefig(os.path.join(dir_out,'grid_est_1d.png'), bbox_inches='tight')


def test_GridEstimation_2d():
    ndim = 2
    bounds = np.array([[-5,5],[-5,5]])

    def ln_prior(x):
        return np.log(1/100)

    def ln_likelihood(x):
        return -(x[0]-1)**2/100 - (x[0]**2-x[1])**2
    
    nbins = [50,40]
    grid_estimator = GridEstimation(ndim, bounds, ln_prior=ln_prior,
        ln_likelihood=ln_likelihood)
    posterior, x_ndim = grid_estimator.run(nbins)
    assert posterior.shape == tuple(nbins)
    assert len(x_ndim) == ndim
    
    for i in range(len(nbins)):
        assert x_ndim[i].shape == (nbins[i],)
    
    grid_size = np.prod((bounds[:,1]-bounds[:,0])/np.array(nbins))
    assert np.abs(np.sum(posterior)*grid_size - 1) < 1e-10

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    posterior = np.where(posterior < 1e-10, np.nan, posterior)
    contour = ax.contour(x_ndim[0], x_ndim[1], np.transpose(posterior), levels=10)
    plt.colorbar(contour, ax=ax)
    ax.set_xlim(bounds[0,0], bounds[0,1])
    ax.set_ylim(bounds[1,0], bounds[1,1])
    ax.set_title(
        'Grid estimation \n prior U[-5,5]x[-5,5] \n likelihood Rosenbrock')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    os.chdir(dir_test)
    if not os.path.exists('temp_bayes_inference'):
        os.mkdir('temp_bayes_inference')
    dir_out = os.path.join(dir_test, 'temp_bayes_inference')
    plt.savefig(os.path.join(dir_out,'grid_est_2d.png'), bbox_inches='tight')


def test_GridEstimation_3d():
    ndim = 3
    bounds = np.array([[-5,5],[-5,5],[-5,5]])

    def ln_pxl(x, mean, cov):
        # uniform prior, multivariate normal likelihood
        return np.log(1/1000) + multivariate_normal.logpdf(x, mean, cov)
    
    kwgs_ln_pxl={'mean':[0,2,-1], 'cov':[[1,0.1,4],[-0.5,2,0.2],[0.5,-2,3]]}
    
    nbins = [30,40,50]
    grid_estimator = GridEstimation(ndim, bounds, ln_pxl=ln_pxl,
        kwgs_ln_pxl=kwgs_ln_pxl)
    posterior, x_ndim = grid_estimator.run(nbins)
    assert posterior.shape == tuple(nbins)
    assert len(x_ndim) == ndim

    for i in range(len(nbins)):
        assert x_ndim[i].shape == (nbins[i],)

    grid_size = np.prod((bounds[:,1]-bounds[:,0])/np.array(nbins))
    assert np.abs(np.sum(posterior)*grid_size - 1) < 1e-10

    # plot
    fig, axes = plt.subplots(3,3,figsize=(12,12))

    for i in range(3):
        indices_except_i = [index for index in [0,1,2] if index != i]
        bounds_except_i = bounds[indices_except_i, :]
        nbins_except_i = np.array(nbins)[indices_except_i]
        posterior_i = np.sum(posterior, axis=tuple(indices_except_i)) * \
            np.prod((bounds_except_i[:,1]-bounds_except_i[:,0])/nbins_except_i)
        assert (np.sum(posterior_i)*(bounds[i,1]-bounds[i,0])/nbins[i] - 1) < \
            1e-10
        axes[i,i].plot(x_ndim[i], posterior_i, 'b-')
    
    for i,j in itertools.product(range(3), range(3)):

        if i != j:
            indices_except_ij = [index for index in [0,1,2] 
                if index != i and index != j]
            bounds_except_ij = bounds[indices_except_ij, :]
            nbins_except_ij = np.array(nbins)[indices_except_ij]
            # shape (ndim[i], ndim[j])
            posterior_ij = \
                np.sum(posterior, axis=tuple(indices_except_ij)) * np.prod(
                (bounds_except_ij[:,1]-bounds_except_ij[:,0])/nbins_except_ij)
            
            if i < j:
                # horizontal axis: xj, vertical axis: xi
                axes[i,j].contour(x_ndim[j], x_ndim[i], posterior_ij, levels=10)
            if i > j:
                axes[i,j].contour(x_ndim[j], x_ndim[i],
                    np.transpose(posterior_ij), levels=10)

            axes[i,j].set_ylim(-5,5)
        
        axes[i,j].set_xlim(-5,5)
        axes[i,j].set_xlabel(f'x{j+1}')
        axes[i,j].set_ylabel(f'x{i+1}')

    axes[0,1].set_title('Grid estimation \n' + 'prior U[-5,5]x[-5,5]x[-5-5] \n' +
        f'likelihood MN(mean={kwgs_ln_pxl["mean"]}, cov={kwgs_ln_pxl["cov"]})')
    
    os.chdir(dir_test)
    if not os.path.exists('temp_bayes_inference'):
        os.mkdir('temp_bayes_inference')
    dir_out = os.path.join(dir_test, 'temp_bayes_inference')
    plt.savefig(os.path.join(dir_out,'grid_est_3d.png'), bbox_inches='tight')


def test_MetropolisHastingsEstimation():
    ndim = 2
    bounds = np.array([[-5,5],[-5,5]])

    def ln_pxl(x):
        return np.log(1/100) -(x[0]-1)**2/100 - (x[0]**2-x[1])**2
    
    mh_estimator = MetropolisHastingsEstimation(ndim, bounds, ln_pxl=ln_pxl)

    init_state = np.array([-4,-4])
    f_sample = multivariate_normal.rvs
    nburn = 100
    nthin = 10
    seed = 1
    kwgs_f_sample = {'random_state': np.random.default_rng(seed)}

    mh_sampler = MetropolisHastings(ndim=ndim, init_state=init_state,
        f_sample=f_sample, bounds=bounds, nburn=nburn, nthin=nthin, seed=seed,
        kwgs_f_sample=kwgs_f_sample)
    
    nsamples = 5000
    mh_samples, mh_accept = mh_estimator.run(nsamples, mh_sampler)

    assert mh_samples.shape == (nsamples, ndim)
    assert mh_accept.shape == (nsamples,)
    print('acceptance ratio: ', np.mean(mh_accept))

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.scatter(mh_samples[:, 0],mh_samples[:,1], s=10, c='r',
        marker='o', alpha=0.1)
    ax.set_xlim(bounds[0,0], bounds[0,1])
    ax.set_ylim(bounds[1,0], bounds[1,1])
    ax.set_title(
        'MH estimation \n prior U[-5,5]x[-5,5] \n likelihood Rosenbrock')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    os.chdir(dir_test)
    if not os.path.exists('temp_bayes_inference'):
        os.mkdir('temp_bayes_inference')
    dir_out = os.path.join(dir_test, 'temp_bayes_inference')
    plt.savefig(os.path.join(dir_out,'mh_est_2d.png'), bbox_inches='tight')