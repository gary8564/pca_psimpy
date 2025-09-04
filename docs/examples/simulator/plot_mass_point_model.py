"""

Mass Point Model
================
"""

# %% md
# 
# This example shows how to simulate the movement of a masspoint on a topography
# using :class:`.MassPointModel`.
#
# 

# %% md
# 
# First, import the class :class:`.MassPointModel` and create an instance by

from psimpy.simulator import MassPointModel

mpm = MassPointModel()

# %% md
# 
# Required inputs for a simulation using :class:`.MassPointModel` include:
#     1. topographic data: digital elevation model (in `ESRI ascii` format)
#     2. friction coefficients: coulomb friction and turbulent friction coefficients
#     3. initial state: initial location and initial velocity of the masspoint
#     4. computational parameters: such as time step, end time, etc.
#
# The synthetic topography ``synthetic_topo.asc`` is used here for illustration.
# It is located at the `tests/data/` folder.

import os
import linecache
import numpy as np

dir_data = os.path.abspath('../../../tests/data/')
elevation = os.path.join(dir_data, 'synthetic_topo.asc')


# %% md
# .. note:: You may need to modify ``dir_data`` according to where you save
#    ``synthetic_topo.asc`` on your local machine.

# %% md
# We can load the elevation data and visulize it. The figure below shows how
# the topography looks like, as well as the initial location of the masspoint
# (noted by the red dot).

header = [linecache.getline(elevation, i) for i in range(1,6)]
header_values = [float(h.split()[-1].strip()) for h in header]
ncols, nrows, xll, yll, cellsize = header_values
ncols = int(ncols)
nrows = int(nrows)

x_values = np.arange(xll, xll+(cellsize*ncols), cellsize)
y_values = np.arange(yll, yll+(cellsize*nrows), cellsize)
        
z_values = np.loadtxt(elevation, skiprows=6)
z_values = np.rot90(np.transpose(z_values))

# initial location
x0 = 200
y0 = 2000
z0 = z_values[0, int(x0/cellsize)]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10,6), height_ratios=[3,2])

fig0 = ax[0].contourf(x_values, y_values, z_values, levels=20)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('synthetic topography')
ax[0].scatter(x0, y0, s=10, c='r', marker='o')
cbar = plt.colorbar(fig0, ax=ax[0], format='%d', orientation='horizontal',
    fraction=0.1, pad=0.2)
cbar.ax.set_ylabel('z')


ax[1].plot(x_values, z_values[0, :])
ax[1].scatter(x0, z0, s=10, c='r', marker='o')
ax[1].set_xlabel('x')
ax[1].set_ylabel('z')
ax[1].set_xlim(0, 5000)
ax[1].set_title('cross section at any y')

plt.tight_layout()

# %% md
# We set the friction coefficients as

mu = 0.15
xi = 1000

# %% md
# Given above topography, initial location, and friction coefficients, we can
# call the :py:meth:`.MassPointModel.run` method to perform a simulation. Other
# parameters are set to their default values. (we suppress raised warnings )
# (other parameters are set to their default values).

import warnings

warnings.filterwarnings("ignore")
output = mpm.run(elevation=elevation, coulomb_friction=mu, turbulent_friction=xi, x0=x0, y0=y0)

# %% md
# The simulation returns time history of the mass point's location and velocity.
# Following plots show the simulation results.

fig, ax = plt.subplots(3, 1, figsize=(10,6))

ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('x')
ax[0].set_title("x-t plot")
ax[0].plot(output[:,0], output[:,1])

ax[1].set_xlabel('time (s)')
ax[1].set_ylabel('velocity')
ax[1].set_title("v-t plot")
ax[1].plot(output[:,0], output[:,5])

ax[2].set_xlabel('x')
ax[2].set_ylabel('velocity')
ax[2].set_title("x-v plot")
ax[2].plot(output[:,1], output[:,5])

plt.tight_layout()
