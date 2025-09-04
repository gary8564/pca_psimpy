"""

ScalarGaSP: GP emulation for a single-output function
=====================================================
"""

# %% md
# 
# This example shows how to apply Gaussian process emulation to a single-output
# function using class :class:`.ScalarGaSP`.
#
# The task is to build a GP emulator for the function :math:`y = x * sin(x)`
# based on a few number of training data.

# %% md
# 
# First, import the class :class:`.ScalarGaSP` by

from psimpy.emulator import ScalarGaSP

# %% md
# 
# Then, create an instance of :class:`.ScalarGaSP`.  The parameter ``ndim``
# (dimension of function input ``x``) must be specified. Optional parameters, such
# as ``method``, ``kernel_type``, etc., can be set up if desired. Here, we leave
# all the optional parameters to their default values.

emulator = ScalarGaSP(ndim=1)

# %% md
# 
# Given training input points ``design`` and corresponding output values ``response``,
# the emulator can be trained using :py:meth:`.ScalarGaSP.train`. Below we train
# an emulator using :math:`8` selected points.

import numpy as np

def f(x):
    #return x + 3*np.sin(x/2)
    return x*np.sin(x)

x = np.arange(2,10,1)
y = f(x)

emulator.train(design=x, response=y)

# %% md
# 
# We can validate the performance of the trained emulator using the leave-one-out
# cross validation method :py:meth:`loo_validate()`.

validation = emulator.loo_validate()

# %% md
#
# Let's plot emulator predictions vs actual outputs. The error bar indicates the
# standard deviation.

import matplotlib.pyplot as plt

fig , ax = plt.subplots(figsize=(4,4))

ax.set_xlabel('Actual y')
ax.set_ylabel('Emulator-predicted y')
ax.set_xlim(np.min(y)-1,np.max(y)+1)
ax.set_ylim(np.min(y)-1,np.max(y)+1)

_ = ax.plot([np.min(y)-1,np.max(y)+1], [np.min(y)-1,np.max(y)+1])
_ = ax.errorbar(y, validation[:,0], validation[:,1], fmt='.', linestyle='', label='prediction and std')
_ = plt.legend()
plt.tight_layout()


# %% md
#
# With the trained emulator at our deposit, we can use the
# :py:meth:`.ScalarGaSP.predict()` method to make predictions at 
# any arbitrary set of input points (``testing_input``). It should be noted that,
# ``testing_trend`` should be set according to ``trend`` used during emulator
# training. 

testing_input = np.arange(0,10,0.1)
predictions = emulator.predict(testing_input)

plt.plot(testing_input, predictions[:, 0], 'r-', label= "mean")
plt.scatter(x, y, s=15, c='k', label="training data", zorder=3)
plt.plot(testing_input, f(testing_input), 'k:', zorder=2, label="true function")
plt.fill_between(testing_input, predictions[:, 1], predictions[:, 2], alpha=0.3, label="95% CI")
plt.xlabel('x')
plt.ylabel('emulator-predicted y')
plt.xlim(testing_input[0], testing_input[-1])
_ = plt.legend()
plt.tight_layout()

# %% md
#
# We can also draw any number of samples at ``testing_input``` using
# :py:meth:`.ScalarGaSP.sample()` method.

nsamples = 5
samples = emulator.sample(testing_input, nsamples=nsamples)

# sphinx_gallery_thumbnail_number = 3
for i in range(nsamples):
    plt.plot(testing_input, samples[:,i], '--', label=f'sample{i+1}')

plt.scatter(x, y, s=15, c='k', label="training data", zorder=2)
plt.plot(testing_input, f(testing_input), 'k:', zorder=2, label="true function")
plt.fill_between(testing_input, predictions[:, 1], predictions[:, 2], alpha=0.3, label="95% CI")
plt.xlabel('x')
plt.ylabel('emulator-predicted y')
plt.xlim(testing_input[0], testing_input[-1])
_ = plt.legend()
plt.tight_layout()


# %% md
#
# .. tip:: Above example shows how to train a GP emulator based on noise-free training data,
#    which is often the case of emulating a deterministic simulator. If you are dealing
#    with noisy training data, you can
#
#     - set the parameter ``nugget`` to a desired value, or
#     - set ``nugget`` to :math:`0` and ``nugget_est`` to `True`, meaning that ``nugget``
#       is estimated from the noisy training data.