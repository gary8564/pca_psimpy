"""

Saltelli sampling
=================
"""

# %% md
# 
# This example shows how to draw samples using Saltelli sampling. 
# Assume that there is a three-dimensional problem where X, Y, and Z are the
# three random variables.

import numpy as np

ndim = 3
# range of X is 10 to 20, range of Y is 100 to 200, range of Z is 1000 to 2000
bounds = np.array([[10,20], [100,200], [1000,2000]])

# %% md
#
# Given this setting, we can import :class:`.Saltelli`, create an instance, and
# call the :py:meth:`.Saltelli.sample` method to draw required number of samples.

from psimpy.sampler import Saltelli

saltelli_sampler = Saltelli(ndim, bounds, calc_second_order=False)
saltelli_samples = saltelli_sampler.sample(nbase=128)

# %% md
#
# In above codes, we set ``calc_second_order`` to `False`. It means that picked
# samples can be used in following Sobol' analysis to compute first-order and
# total-effect Sobol' indices but not second-order Sobol' indices. It leads to 
# :math:`nbase*(ndim+2)` samples as shown below

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(saltelli_samples[:,0], saltelli_samples[:,1], saltelli_samples[:,2], marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()

print('Number of samples: ', f'{len(saltelli_samples)}')


# %% md
#
# If we want to draw samples which can also be used to compute second-order
# Sobol' indices, we need to set ``calc_second_order`` to `True`.
# It leads to :math:`nbase*(2*ndim+2)` samples.

saltelli_sampler = Saltelli(ndim, bounds, calc_second_order=True)
saltelli_samples = saltelli_sampler.sample(nbase=128)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(saltelli_samples[:,0], saltelli_samples[:,1], saltelli_samples[:,2], marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()

print('Number of samples: ', f'{len(saltelli_samples)}')


# %% md
# .. note:: If one has a two-dimensional problem, there is no need to set
#     ``calc_second_order`` to `True`. The reason is that the second-order Sobol'
#     index can be directly computed based on the first-order and total-effect
#     Sobol' index in that case.





