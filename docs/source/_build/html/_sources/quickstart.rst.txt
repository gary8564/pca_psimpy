
Getting Started
===============

Overview
--------

PSimPy (Predictive and probabilistic simulation with Python) implements
a Gaussian process emulation-based framework that enables systematically and
efficiently performing uncertainty-related analyses of physics-based
models, which are often computationlly expensive. Examples are variance-based
global sensitvity analysis, uncertainty quantification, and parameter
calibration.

Prerequisites
-------------

Before installing and using ``PSimPy``, please ensure that you have the
following prerequisites: (Please note that we will cover number 1 to 3 in our
recommended installation method `Installation in a Conda Environment`_.)

#.  **Python 3.9 or later**: Make sure you have Python installed on your system. You can download the latest version of Python from the official website: `Python Downloads <https://www.python.org/downloads/>`_
#.  **R Installed and Added to the PATH Environment Variable**:

    *  Install R from the official `R Project <https://www.r-project.org/>`_ website.
    *  Add R to your system's PATH environment variable. This step is crucial for enabling communication between Python and R.

#.  (Optional) **RobustGaSP - R package**: The emulator module, ``robustgasp.py``, relies on the R package `RobustGaSP <https://cran.r-project.org/web/packages/RobustGaSP/index.html>`__. This has also been initegrated with other PSimPy modules, such as ``active_learning.py``. In order to utilize these modules, make sure to install the R package `RobustGaSP <https://cran.r-project.org/web/packages/RobustGaSP/index.html>`__ first.
#.  (Optional) **r.avaflow - Mass Flow Simulation Tool**: ``PSimPy`` includes a simulator module, ``ravaflow3G.py``, that interfaces with the open source software `r.avaflow 3G <https://www.landslidemodels.org/r.avaflow/>`_. If you intend to use this module, please refer to the official documentation of `r.avaflow 3G <https://www.landslidemodels.org/r.avaflow/>`_ to for installation guide.

Installation
------------

``PSimPy`` can be installed using ``pip``::

    pip install psimpy

This command will install the package along with its dependencies.

..  _Installation in a Conda Environment:

Installation in a Conda Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommond you to install ``PSimPy`` in a virtual environment such as a
``conda`` environment. In this section, we will ceate a ``conda`` environment 
with prerequisites (number 1 to 3), and install python in this environment. You 
may want to first install `Anaconda <https://docs.anaconda.com/free/anaconda/>`_
or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ if you
haven't. The steps afterwards are as follows:

#.  Create a conda environment with Python and R, and RobustGaSP, and activate the environment::

        $ conda create --name your_env_name python r-base conda-forge::r-robustgasp
        $ conda activate your_env_name

#.  Install ``PSimPy`` using ``pip`` in your conda environment::

        $ pip install psimpy

Now you should have ``PSimPy`` and its dependencies successfully installed in
your conda environment. You can use it in the Python terminal or in your Python
IDE.

**Quick Note on R_HOME in Conda Environments:**

If you're running PSimPy in a conda environment without a predefined R_HOME 
variable, we automatically set it to the default R installation path of the 
active conda environment. This ensures PSimPy works smoothly with R without 
needing manual setup. If you prefer setting R_HOME yourself, please define it 
before starting PSimPy to use a custom R environment.
