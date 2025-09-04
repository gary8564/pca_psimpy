"""

Metropolis Hastings sampling
============================
"""

# %% md
# 
# This example shows how to draw samples using Metropolis Hastings sampling.
# The target probability distribution is
#
# :math:`p(\mathbf{x})=p(x_1, x_2) \propto \exp \left(-\frac{1}{100}\left(x_1-1\right)^2-\left(x_1^2-x_2\right)^2\right)`
#
# where :math:`x_1 \in [-5,5]` and :math:`x_2 \in [-5,5]`.
#
# It should be noted that the right hand side of the equation is an unnormalized
# probability density function since its integral is not equal to :math:`1`.
# This can happen, for example, when the normalization constant is unknown or difficult
# to compute.
# 
# We can define the target probability distribution in Python as follows:
# 

import numpy as np

def target_dist(x):
    if (x[0]>=-5 and x[0]<=5) and (x[1]>=-5 and x[1]<=5):
        return np.exp(-0.01*(x[0]-1)**2 - (x[0]**2-x[1])**2)
    else:
        return 0

# %% md
# 
# The figure below shows how the target distribution looks like.

import matplotlib.pyplot as plt
import itertools

x1_values = np.linspace(-5,5.1,100)
x2_values = np.linspace(-5,5.1,100)

target_values = np.zeros((100, 100))
for i, j in itertools.product(range(100), range(100)):
    x1 = x1_values[i]
    x2 = x2_values[j]
    target_values[i,j] = target_dist(np.array([x1, x2]))

fig , ax = plt.subplots(figsize=(4,4))

ax.contourf(x1_values, x2_values, np.transpose(target_values), levels=10, cmap='Blues')

ax.set_title('(unnormalized) target distribution')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
plt.tight_layout()


# %% md
# 
# To perform Metropolis Hastings sampling, we need to choose a proposal distribution
# which can be used to determine a new state ``x'`` given a current state ``x`` at
# each iteration. This is defined by the parameter ``f_sample`` which should be a
# function. A usual choice is to choose the new state from a Gaussian distribution
# centered at the current state.
#

from scipy.stats import multivariate_normal

f_sample = multivariate_normal.rvs
# make the samples reproducible
kwgs_f_sample = {'random_state': np.random.default_rng(seed=1)}

# %% md
# 
# Then, we create an instance of :class:`.MetropolisHastings` class.

from psimpy.sampler.metropolis_hastings import MetropolisHastings

mh_sampler = MetropolisHastings(ndim=2, init_state=np.array([-4,-4]),
    f_sample=f_sample, target=target_dist, nburn=1000, nthin=5,
    seed=1, kwgs_f_sample=kwgs_f_sample)

# %% md
# 
# Next, we call the :py:meth:`.MetropolisHastings.sample` method to draw required 
# number of samples.

mh_samples, mh_accept = mh_sampler.sample(nsamples=5000)

print("Acceptance ratio: ", np.mean(mh_accept))


# %% md
# 
# The following figure shows how the samples look like.

import matplotlib.pyplot as plt
import itertools

# sphinx_gallery_thumbnail_number = 2
fig , ax = plt.subplots(figsize=(4,4))

ax.contourf(x1_values, x2_values, np.transpose(target_values), levels=10, cmap='Blues')
ax.scatter(mh_samples[:,0], mh_samples[:,1], s=5, c='r', marker='o',alpha=0.05)

ax.set_title('Metropolis Hastings samples')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
plt.tight_layout()



