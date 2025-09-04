Sobol' Sensitivity Analysis
===========================

This module is used to compute Sobol' indices given model (e.g. simulator or emulator)
outputs evaluated at chosen Saltelli samples (see the :class:`.Saltelli` class).
Detailed description of the theory can be found in :cite:t:`Saltelli2002` and :cite:t:`Saltelli2010`.
The :class:`.SobolAnalyze` class relies on the Python package `SALib` :cite:p:`Herman2017`.

SobolAnalyze Class
------------------

The :class:`.SobolAnalyze` class is imported by::
    
    from psimpy.sensitivity.sobol import SobolAnalyze

Methods
^^^^^^^
.. autoclass:: psimpy.sensitivity.sobol.SobolAnalyze
    :members: run