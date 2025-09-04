Ravaflow3G Mixture Model
========================

`r.avaflow 3G` is a GIS-supported open source software for mass flow modeling.
It is developed by :cite:t:`Mergili2017`. For more information see its official
user manual `here <https://www.landslidemodels.org/r.avaflow/>`_. 

In :py:mod:`PSimPy.simulator.ravaflow3G`, we have implemented class 
:class:`.Ravaflow3GMixture`. It provides a Python interface to directly run
the `Voellmy-type shallow flow model` of `r.avaflow 3G` from within Python.
For detailed theory of `Voellmy-type shallow flow model`, please refer to
:cite:t:`Christen2010` and :cite:t:`Fischer2012`.

Please note that the :py:mod:`PSimPy.simulator.ravaflow24` module corresponding to `r.avaflow 2.4` has been deprecated.

Ravaflow3GMixture Class
-----------------------

The :class:`.Ravaflow3GMixture` class is imported by::
    
    from psimpy.simulator.ravaflow3G import Ravaflow3GMixture

Methods
^^^^^^^
.. autoclass:: psimpy.simulator.ravaflow3G.Ravaflow3GMixture
    :members: preprocess, run, extract_impact_area, extract_qoi_max, extract_qoi_max_loc