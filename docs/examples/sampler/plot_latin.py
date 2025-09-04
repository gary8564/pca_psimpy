"""

Latin hypercube sampling
========================
"""

# %% md
# 
# This example shows how to draw samples using Latin hypercube sampling.
#
# 

# %% md
# 
# For the illustration purpose, let's have a look at a two-dimensional example
# where we have two random variables X and Y. Each is uniformly distributed in its
# range.

import numpy as np

ndim = 2
# range of X is 10 to 20, range of Y is -10 to 0
bounds = np.array([[10,20], [-10,0]])

# %% md
#
# Given this setting, we can import :class:`.Latin`, create an instance, and
# call the :py:meth:`.Latin.sample` method to draw required number of samples

from psimpy.sampler import LHS

# setting seed leads to same samples every time when the codes are run
lhs_sampler = LHS(ndim, bounds, seed=10)
lhs_samples = lhs_sampler.sample(nsamples=5)

# %% md
#
# The samples are plotted in the following figure.

import matplotlib.pyplot as plt

fig , ax = plt.subplots(figsize=(6,4))

ax.scatter(lhs_samples[:,0], lhs_samples[:,1], s=10, c='blue', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("Latin hypercube samples (criterion='random')")
_ = ax.grid(visible=True, which='major', axis='both')


# %% md
#
# There are different criterions to pick samples in each hypercube. The default
# is `random`, as used above. Other options are `center` and `maximin`. For instance,
# we can use the `center` criterion to draw :math:`5` samples as follows:

lhs_sampler = LHS(ndim, bounds, criterion='center', seed=10)
lhs_samples = lhs_sampler.sample(nsamples=5)

fig , ax = plt.subplots(figsize=(6,4))

ax.scatter(lhs_samples[:,0], lhs_samples[:,1], s=10, c='blue', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("Latin hypercube samples (criterion='center')")
_ = ax.grid(visible=True, which='major', axis='both')


# %% md
#
# And we can use the `maximin` criterion as follows:

lhs_sampler = LHS(ndim, bounds, criterion='maximin', seed=10, iteration=500)
lhs_samples = lhs_sampler.sample(nsamples=5)

fig , ax = plt.subplots(figsize=(6,4))

ax.scatter(lhs_samples[:,0], lhs_samples[:,1], s=10, c='blue', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("Latin hypercube samples (criterion='maximin')")
_ = ax.grid(visible=True, which='major', axis='both')