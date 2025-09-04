"""

Bayesian inference
==================
"""

# %% md
# 
# This example shows how to perform Bayesian inference given the uniform prior
#
# :math:`p(\mathbf{x})=p(x_1,x_2)=0.01`
# 
# where :math:`x_i \in [-5,5], i=1,2`, and likelihood
#
# :math:`L(\mathbf{x}|\mathbf{d})=\exp \left(-\frac{1}{100}\left(x_1-1\right)^2-\left(x_1^2-x_2\right)^2\right)`.
#

import numpy as np

ndim = 2
bounds = np.array([[-5,5],[-5,5]])

def prior(x):
        return 0.01

def likelihood(x):
        return np.exp(-(x[0]-1)**2/100 - (x[0]**2-x[1])**2)

# %% md
#
# To estimate the posterior using grid estimation, we need to import the
# :class:`.GridEstimation` class, create an instance, and call the 
# :py:meth:`.GridEstimation.run` method.

from psimpy.inference.bayes_inference import GridEstimation

grid_estimator = GridEstimation(ndim, bounds, prior, likelihood)
posterior, x_ndim = grid_estimator.run(nbins=[50,40])

# %% md
#
# The following figure plots the estimated posterior.
# 

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(6,4))

# mask insignificant values
posterior = np.where(posterior < 1e-10, np.nan, posterior)

contour = ax.contour(x_ndim[0], x_ndim[1], np.transpose(posterior), levels=10)
plt.colorbar(contour, ax=ax)
ax.set_xlim(bounds[0,0], bounds[0,1])
ax.set_ylim(bounds[1,0], bounds[1,1])
ax.set_title('Grid estimation')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.tight_layout()

# %% md
#
# To estimate the posterior using Metropolis Hastings estimation, we need to import
# the :class:`.MetropolisHastingsEstimation` class, create an instance, and call the 
# :py:meth:`.MetropolisHastingsEstimation.run` method. The 
# :py:meth:`.MetropolisHastingsEstimation.run` method has a parameter, ``mh_sampler``,
# which takes an instance of :class:`.MetropolisHastings` as argument.

from psimpy.inference.bayes_inference import MetropolisHastingsEstimation

mh_estimator = MetropolisHastingsEstimation(ndim, bounds, prior, likelihood)

# create a mh_sampler
from psimpy.sampler.metropolis_hastings import MetropolisHastings
from scipy.stats import multivariate_normal

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


# %% md
#
# The following figure plots the samples drawn from the unnormalized posterior,
# which can be used to estimate the posterior and its poperties.
# 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(5,4))

ax.scatter(mh_samples[:,0], mh_samples[:,1], s=10, c='r', marker='o', alpha=0.1)
ax.set_xlim(bounds[0,0], bounds[0,1])
ax.set_ylim(bounds[1,0], bounds[1,1])
ax.set_title('MH estimation')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.tight_layout()


# %% md
# .. note:: Besides ``prior`` and ``likelihood``, one can also instantiate
#     the :class:`.MetropolisHastingsEstimation` class with
#
#      - ``ln_prior`` and ``ln_likelihood``: Natural logarithm of ``prior`` and ``likelihood``.
#      - ``ln_pxl``: Natural logarithm of the product of ``prior`` and ``likelihood``.
#