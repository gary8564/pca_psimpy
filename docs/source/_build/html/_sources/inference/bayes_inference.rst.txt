Bayes Inference
===============

Bayesian inference is a statistical method for making probabilistic inference about
unknown parameters based on observed data and prior knowledge. The update from
prior knowledge to posterior in light of observed data is based on Bayes' theorem:

.. math:: p(\mathbf{x} \mid \mathbf{d}) = \frac{L(\mathbf{x} \mid \mathbf{d}) p(\mathbf{x})}
    {\int L(\mathbf{x} \mid \mathbf{d}) p(\mathbf{x}) d \mathbf{x}}
    :label: bayes

where :math:`\mathbf{x}` represents the collection of unknown parameters and
:math:`\mathbf{d}` represents the collection of observed data. The prior probability
distribution, :math:`p(\mathbf{x})`, represents the degree of belief in the parameters before
any data is observed. The likelihood function, :math:`L(\mathbf{x} \mid \mathbf{d})`, represents
the probability of observing the data given the parameters. The posterior distribution,
:math:`p(\mathbf{x} \mid \mathbf{d})`, is obtained by multiplying the prior probability
distribution by the likelihood function and then normalizing the result. Bayesian inference
allows for incorporating subjective prior beliefs, which can be updated as new data becomes available.

For many real world problems, it is hardly possible to analytically compute the posterior due to
the complexity of the denominator in equation :eq:`bayes`, namely the nomalizing constant. In this
module, two numerical approximations are implemented: grid estimation and Metropolis Hastings
estimation.

In grid estimation, the denominator in equation :eq:`bayes` is approximated by numerical integration
on a regular grid and the posterior value at each grid point is computed, as shown in equation
:eq:`grid_estimation`. The number of grid points increases dramatically with the increase of the number
of unknown parameters. Grid estimation is therefore limited to low-dimensional problems.

.. math:: p(\mathbf{x} \mid \mathbf{d}) \approx \frac{L(\mathbf{x} \mid \mathbf{d}) p(\mathbf{x})}
    {\sum_{i=1}^N L\left(\mathbf{x}_i \mid \mathbf{d}\right) p\left(\mathbf{x}_i\right) \Delta \mathbf{x}_i}
    :label: grid_estimation

Metropolis Hastings estimation directly draw samples from the unnormalized posterior distribution, namely
the numerator of equation :eq:`bayes`. The samples are then used to estimate properties of the posterior
distribution, like the mean and variance, or to estimate the posterior distribution.


GridEstimation Class
--------------------

The :class:`.GridEstimation` class is imported by::
    
    from psimpy.inference.bayes_inference import GridEstimation

Methods
^^^^^^^
.. autoclass:: psimpy.inference.bayes_inference.GridEstimation
    :members: run


MetropolisHastingsEstimation Class
----------------------------------

The :class:`.MetropolisHastingsEstimation` class is imported by::
    
    from psimpy.inference.bayes_inference import MetropolisHastingsEstimation

Methods
^^^^^^^
.. autoclass:: psimpy.inference.bayes_inference.MetropolisHastingsEstimation
    :members: run
