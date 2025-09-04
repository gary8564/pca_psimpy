Saltelli Sampling
=================

Saltelli sampling is a method to draw samples for the purpose of Sobol'
sensitvity analysis (variance-based sensitivity analysis). Detailed description
of the theory can be found in :cite:t:`Saltelli2002` and :cite:t:`Saltelli2010`.
The :class:`.Saltelli` class relies on the Python package `SALib` :cite:p:`Herman2017`.

Saltelli Class
--------------

The :class:`.Saltelli` class is imported by::
    
    from psimpy.sampler.saltelli import Saltelli

Methods
^^^^^^^
.. autoclass:: psimpy.sampler.saltelli.Saltelli
    :members: sample