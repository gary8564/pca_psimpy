Active Learning
===============

Active learning is a machine learning technique that involves selecting the most
informative data points for the purpose of training an emulator. The idea behind
active learning is to reduce the amount of training data required for a machine
learning model to achieve a certain level of accuracy. This is achieved by iteratively
choosing a new data point that is expected to be the most informative.

In this module, the :class:`.ActiveLearning` class is implemented to actively build a
Gaussian process emulator for the natural logarithm of the unnormalized posterior in
Bayesian inference. It is supposed to facilitate efficient parameter calibration of
computationally expensive simulators. For detailed theories, please refer to 
:cite:t:`Wang2018`, :cite:t:`Kandasamy2017`, and :cite:t:`Zhao2022`.


ActiveLearning Class
--------------------

The :class:`.ActiveLearning` class is imported by::
    
    from psimpy.inference.active_learning import ActiveLearning

Methods
^^^^^^^
.. autoclass:: psimpy.inference.active_learning.ActiveLearning
    :members: initial_simulation, iterative_emulation, approx_ln_pxl