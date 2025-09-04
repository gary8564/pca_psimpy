"""

Active learning
===============
"""

# %% md
# 
# This example shows how to use the :class:`.ActiveLearning` class to iteratively
# build a Gaussian process emulator for an unnormalized posterior involving a
# simulator. It should be noted that this example is only for illustration
# purpose, rather than a real case. For simplicity, required arguments for
# :class:`.ActiveLearning` (including `simulator`, `likelihood`, `data`, etc.) are purely
# made up. For a realistic case of active learning, one can refer to :cite:t:`Zhao2022`.
#
# First, we define the simulator, prior distribution of its variable parameters,
# observed data, and likelihood function. They basically define the Bayesian
# inference problem. 

import numpy as np

ndim = 2 # dimension of variable parameters of the simulator
bounds = bounds = np.array([[-5,5],[-5,5]]) # bounds of variable parameters
data = np.array([1,0])

def simulator(x1, x2):
    """Simulator y=f(x)."""
    y1, y2 = x1, x2
    return np.array([y1, y2])

def prior(x):
    """Uniform prior."""
    return 1/(10*10)

def likelihood(y, data):
    """Likelihood function L(y,data)."""
    return np.exp(-(y[0]-data[0])**2/100 - (y[0]**2-y[1]-data[1])**2)

# %% md
#
# Imagine that the simulator is a complex solver. It is not computationally feasible
# to compute the posterior distribution of the variable parameters using grid estimation
# or Metropolis Hastings estimation. This is because they require evaluating the likelihood
# many times which essentially leads to many evaluations of the simulator. Therefore, we
# resort to use active learning to build a Gaussian process emulator for the unnormalized
# posterior (prior times likelihood) based on a small number of evaluations of the simulator.
# The the posterior can be estimated using the emulator.
#
# To do so, we need to pass arguments to following parameters of the :class:`.ActiveLearning` class:
#
#   - run_sim_obj : instance of class :class:`.RunSimulator`. It carries information on how
#     to run the simulator.
#   - lhs_sampler : instance of class :class:`.LHS`. It is used to draw initial samples to run
#     simulations in order to train an inital Gaussian process emulator.
#   - scalar_gasp : instance of class :class:`.ScalarGaSP`. It sets up the emulator structure.
#

from psimpy.simulator import RunSimulator
from psimpy.sampler import LHS
from psimpy.emulator import ScalarGaSP

run_simulator = RunSimulator(simulator, var_inp_parameter=['x1','x2'])
lhs_sampler = LHS(ndim=ndim, bounds=bounds, seed=1)
scalar_gasp = ScalarGaSP(ndim=ndim)

# %%md
#
# Next, we create an object of the :class:`.ActiveLearning` class by

from psimpy.inference import ActiveLearning

active_learner = ActiveLearning(ndim, bounds, data, run_simulator, prior, likelihood,
    lhs_sampler, scalar_gasp)

# %%md
#
# Then we can call the :py:meth:`.ActiveLearning.initial_simulation` method to run initial
# simulations and call the :py:meth:`.ActiveLearning.iterative_emulation` method to
# iteratively run new simulation and build emulator. Here we allocate 40 simulations for
# initial emulator training and 60 simulations for adaptive training.
#
n0 = 40
niter = 60

init_var_samples, init_sim_outputs = active_learner.initial_simulation(
    n0, mode='parallel', max_workers=4)

var_samples, _, _ = active_learner.iterative_emulation(
    n0, init_var_samples, init_sim_outputs, niter=niter)

# %%md
#
# Once the active learning process is finished, we obtain the final emulator for the
# logarithm of the unnormalized posterior, which is given by the
# :py:meth:`.ActiveLearning.approx_ln_pxl` method.
# 
# We can then estimate the posterior using grid estimation or Metropolis Hastings
# estimation based on the emulator. An example is as follows. The contour plot shows
# the estimated posterior. 
#
from psimpy.inference import GridEstimation
import matplotlib.pyplot as plt

grid_estimator = GridEstimation(ndim, bounds, ln_pxl=active_learner.approx_ln_pxl)
posterior, x_ndim = grid_estimator.run(nbins=50)

fig, ax = plt.subplots(1,1,figsize=(5,4))

# initial training points
ax.scatter(init_var_samples[:,0], init_var_samples[:,1], s=10, c='r', marker='o',
    zorder=1, alpha=0.8, label='initial training points')
# actively picked training points
ax.scatter(var_samples[n0:,0], var_samples[n0:,1], s=15, c='k', marker='+',
    zorder=2, alpha=0.8, label='iterative training points')

# estimated posterior based on the final emulator
posterior = np.where(posterior < 1e-10, np.nan, posterior)
contour = ax.contour(x_ndim[0], x_ndim[1], np.transpose(posterior), levels=10, zorder=0)
plt.colorbar(contour, ax=ax)

ax.legend()
ax.set_title('Active learning')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
plt.tight_layout()