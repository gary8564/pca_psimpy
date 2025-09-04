"""

PPGaSP: GP emulation for multi-output functions
===============================================
"""


# %% md
# 
# This example shows how to apply Gaussian process emulation to a multi-output
# function using class :class:`.PPGaSP`.
#
# The multi-output function that we are going to look at is the `DIAMOND`
# (diplomatic and military operations in a non-warfighting domain) computer model.
# It is used as a testbed to illustrate the `PP GaSP` emulator in the `R` package
# `RobustGaSP`, see :cite:t:`Gu2019` for more detail.
#
# The simulator has :math:`13` input parameters and :math:`5` outputs. Namely,
# :math:`\mathbf{y}=f(\mathbf{x})` where :math:`\mathbf{x}=(x_1,\ldots,x_{13})^T`
# and :math:`\mathbf{y}=(y_1,\ldots,y_5)^T`.
#
# The training and testing data are provided in the folder '.../tests/data/'.
# We first load the training and testing data. 

import numpy as np
import os

dir_data = os.path.abspath('../../../tests/data/')

humanityX = np.genfromtxt(os.path.join(dir_data, 'humanityX.csv'), delimiter=',')
humanityY = np.genfromtxt(os.path.join(dir_data, 'humanityY.csv'), delimiter=',')
print(f"Number of training data points: ", humanityX.shape[0])
print(f"Input dimension: ", humanityX.shape[1])
print(f"Output dimension: ", humanityY.shape[1])

humanityXt = np.genfromtxt(os.path.join(dir_data, 'humanityXt.csv'), delimiter=',')
humanityYt = np.genfromtxt(os.path.join(dir_data, 'humanityYt.csv'), delimiter=',')
print(f"Number of testing data points: ", humanityXt.shape[0])

# %% md
# .. note:: You may need to modify ``dir_data`` according to where you save them
#    on your local machine.

# %% md
# ``humanityX`` and ``humanitY`` are the training data, corresponding to ``design``
# and ``response`` respectively. ``humanityXt`` are testing input data, at which
# we are going to make predictions once the emulator is trained. ``humanityYt``
# are the true outputs at ``humanityXt``, which is then used to validate the
# performance of the trained emulator.

# %% md
# 
# To build a `PP GaSP` emulator for the above simulator,  first import class
# :class:`.PPGaSP` by

from psimpy.emulator import PPGaSP

# %% md
# 
# Then, create an instance of :class:`.PPGaSP`.  The parameter ``ndim``
# (dimension of function input ``x``) must be specified. Optional parameters, such
# as ``method``, ``kernel_type``, etc., can be set up if desired. Here, we leave
# all the optional parameters to their default values.

emulator = PPGaSP(ndim=humanityX.shape[1])


# %% md
# 
# Next, we train the `PP GaSP` emulator based on the training data using
# :py:meth:`.PPGaSP.train` method.

emulator.train(design=humanityX, response=humanityY)

# %% md
#
# With the trained emulator, we can make predictions for any arbitrary set of
# input points using :py:meth:`PPGaSP.predict` method.
# Here, we make predictions at testing input points ``humanityXt``.

predictions = emulator.predict(humanityXt)

# %% md
# 
# We can validate the performance of the trained emulator based on the true outputs
# ``humanityYt``.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(5, 1, figsize=(6,15))

for i in range(humanityY.shape[1]):

    ax[i].set_xlabel(f'Actual $y_{i+1}$')
    ax[i].set_ylabel(f'Emulator-predicted $y_{i+1}$')
    ax[i].set_xlim(np.min(humanityYt[:,i]),np.max(humanityYt[:,i]))
    ax[i].set_ylim(np.min(humanityYt[:,i]),np.max(humanityYt[:,i]))

    _ = ax[i].plot([np.min(humanityYt[:,i]),np.max(humanityYt[:,i])], [np.min(humanityYt[:,i]),np.max(humanityYt[:,i])])
    _ = ax[i].errorbar(humanityYt[:,i], predictions[:,i,0], predictions[:,i,3], fmt='.', linestyle='', label='prediction and std')
    _ = ax[i].legend()

plt.tight_layout()


# %%md
#
# We can also draw any number of samples at testing input ``humanityXt`` using 
# :py:meth:`.PPGaSPsample()` method.

samples = emulator.sample(humanityXt, nsamples=10)
print("Shape of samples: ", samples.shape)