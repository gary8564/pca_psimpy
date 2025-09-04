*********
Inference
*********

This module hosts functionality for inference methods.

Currently implemented classes are:

* :class:`.ActiveLearning`: Construct a Gaussian process emulator for an unnormalized posterior. 

* :class:`.GridEstimation`: Estimate a posterior distribution by grid estimation.

* :class:`.MetropolisHastingsEstimation`: Estimate a posterior by Metropolis Hastings estimation.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Inference

    Active Learning <active_learning>
    Bayes Inference <bayes_inference>