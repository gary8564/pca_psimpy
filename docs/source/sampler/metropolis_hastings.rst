Metropolis Hastings Sampling
============================

Metropolis Hastings sampling is a widely used Markov Chain Monte Carlo (MCMC)
algorithm for generating a sequence of samples from a target probability
distribution where direct sampling is difficult. The samples can be used to
estimate properties of the target distribution, like its mean, variance, and higher
moments. The samples can also be used to approximate the target distribution,
for example via a histogram. The algorithm works by constructing a Markov chain
with a stationary distribution equal to the target distribution. 

Steps of the algorithm are as follows:

  1. Choose an initial state for the Markov chain, usually from the target distribution.
  2. At each iteration, propose a new state by sampling from a proposal distribution.
  3. Calculate the acceptance probability.
  4. Accept or reject the proposed state based on the acceptance probability.
  5. Repeat steps 2-4 for a large number of iterations to obtain a sequence of samples.
  6. Apply "burn-in" and "thining".

Final samples from the Markov chain can then be used to estimate the target
distribution or its properties.
       

MetropolisHastings Class
------------------------

The :class:`.MetropolisHastings` class is imported by::
    
    from psimpy.sampler.metropolis_hastings import MetropolisHastings

Methods
^^^^^^^
.. autoclass:: psimpy.sampler.metropolis_hastings.MetropolisHastings
    :members: sample