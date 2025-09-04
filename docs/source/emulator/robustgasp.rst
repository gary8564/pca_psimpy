RobustGaSP
==========

`RobustGaSP`, standing for `Robust Gaussian Stochastic Process Emulation`, is a
`R` package developed by :cite:t:`Gu2019`. It implements robust methods to
estimate the unknown parameters which can greatly improve predictive performance
of the built emulator. In addition, it implements the `PP GaSP` (parallel partial
Gaussian stochastic process) emulator for simulators which return multiple values
as output, see :cite:t:`Gu2016` for more detail.

In :py:mod:`PSimPy.emulator.robustgasp`, we have implemented class
:class:`.ScalarGaSP` and  class :class:`.PPGaSP` based on `RobustGaSP` and
`rpy2`, which allows us to use `RobustGaSP` directly from within `Python`.

Theory recap of GP emulation
----------------------------

A simulator represents a mapping from an input space to an output space.
Gaussian process emulation treats a computationally expensive simulator as an
unknown function from a Bayesian perspective: the prior belief of the simulator,
namely a Gaussian process, is updated based on a modest number of simulation runs,
leading to a posterior which can be evaluated much faster than the simulator and
can then be used for computationally demanding analyses :cite:p:`Zhao2021a`. A recap
of Gaussian process emulation is provided below, which is adapted from `Chapter 3`
of :cite:t:`Zhao2021b`.

GP prior
^^^^^^^^

Let :math:`f(\cdot)` denote a simulator that one wants to approximate. It
defines a mapping from a :math:`p`-dimensional input
:math:`\mathbf{x}=(x_1,\ldots,x_p)^T \in \mathcal{X} \subset{\mathbb{R}^p}`
to a scalar output :math:`y \in \mathbb{R}`. 

From the Bayesian perspective, the prior belief of the simulator is modeled by
a Gaussian process, namely

.. math:: f(\cdot) \sim \mathcal{GP}(m(\cdot), C(\cdot,\cdot)).
    :label: GP

The Gaussian process is fully specified by its mean function :math:`m(\cdot)`
and covariance function :math:`C(\cdot,\cdot)`. The mean function is usually
modeled by parametric regression

.. math:: m(\mathbf{x})=\mathbf{h}^T(\mathbf{x}) \boldsymbol{\beta}=\sum_{t=1}^{q} h_{t}(\mathbf{x}) \beta_{t},
    :label: GPmean

with :math:`q`-dimensional vectors
:math:`\mathbf{h}(\mathbf{x})=\left(h_{1}(\mathbf{x}), h_{2}(\mathbf{x}), \ldots, h_{q}(\mathbf{x})\right)^T`
and :math:`\boldsymbol{\beta}=\left(\beta_{1}, \beta_{2}, \ldots, \beta_{q}\right)^T`
denoting the chosen basis functions and unknown regression parameters respectively.
The basis functions are commonly chosen as constant :math:`\mathbf{h}(\mathbf{x})=1`
or some prescribed polynomials like :math:`\mathbf{h}(\mathbf{x})=(1,x_1,\ldots,x_p)^T`.

The covariance function is often assumed to be stationary and typically has a
separable form of

.. math::  C\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\sigma^2 c\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\sigma^2 \prod_{l=1}^{p} c_{l}\left(x_{i l}, x_{j l}\right),
    :label: GPcovariance

where :math:`c\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)` is the correlation
function, and :math:`c_{l}\left(x_{i l}, x_{j l}\right)` corresponds to the
correlation between the :math:`l`-th component of :math:`\mathbf{x}_i`` and
the counterpart of :math:`\mathbf{x}_j`. Various correlation functions can be
chosen for :math:`c_{l}\left(x_{i l}, x_{j l}\right)`. In what follows a specific
correlation function from the Matern family is chosen to illustrate the theory.
It should be noted that it can be replaced by any other correlation function
without loss of generality.

.. math:: c_{l}\left(x_{i l}, x_{j l}\right) = \left(1+\frac{\sqrt{5}||x_{il}-x_{jl}||}{\psi_l}+\frac{5||x_{il}-x_{jl}||^2}{3\psi_{l}^{2}} \right) \exp{\left(-\frac{\sqrt{5}||x_{il}-x_{jl}||}{\psi_l}\right)},
    :label: GPcorrelation

where :math:`\psi_l` is the correlation length parameter in the :math:`l`-th
dimension. The unknowns in the covariance function
:math:`C\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)` are the variance
:math:`\sigma^2` and the :math:`p` correlation length parameters
:math:`\boldsymbol{\psi}=(\psi_1,\ldots,\psi_p)^T`.

GP posterior
^^^^^^^^^^^^

Equations :eq:`GP` to :eq:`GPcorrelation` encode prior knowledge of the simulator
:math:`f(\cdot)`. From the Bayesian viewpoint, this prior knowledge can be updated
to a posterior based on a limited number of training data. Training data here refer
to :math:`n^{tr}` pairs of input-output data of :math:`f(\cdot)`.

Let :math:`\mathbf{x}^{tr}:=\{\mathbf{x}_i\}_{i=1}^{n^{tr}}` and 
:math:`\mathbf{y}^{tr}:=\{f(\mathbf{x}_i)\}_{i=1}^{n^{tr}}` denote the training
inputs and outputs respectively. Since any finite number of random variables
from a Gaussian process are jointly Gaussian distributed, the joint distribution
of the :math:`n^{tr}` outputs :math:`\mathbf{y}^{tr}` follow a
:math:`n^{tr}`-dimensional Gaussian distribution

.. math:: p(\mathbf{y}^{tr} | \boldsymbol{\beta}, \sigma^{2}, \boldsymbol{\psi})  \sim \mathcal{N}_{n^{tr}}\left(\mathbf{H}\boldsymbol{\beta}, \sigma^{2} \mathbf{R}\right),
    :label: GPlikelihood1

where :math:`\mathbf{H}=\left[\mathbf{h}(\mathbf{x}_{1}), \ldots, \mathbf{h}(\mathbf{x}_{n^{tr}}) \right]^T`
is the :math:`n^{tr} \times q` basis design matrix and :math:`\mathbf{R}` is the
:math:`n^{tr} \times n^{tr}` correlation matrix with :math:`(i,j)` element
:math:`c(\mathbf{x}_{i}, \mathbf{x}_{j})`. Equation :eq:`GPlikelihood1` is known
as the likelihood function.

Similarly, the joint distribution of the function output :math:`y^*` at any
untried input :math:`\mathbf{x}^*` and :math:`\mathbf{y}^{tr}` follows a
:math:`(n^{tr}+1)`-dimensional Gaussian distribution

.. math:: p(y^*, \mathbf{y}^{tr} \mid \boldsymbol{\beta}, \sigma^{2}, \boldsymbol{\psi}) \sim \mathcal{N}_{n^{tr}+1}\left(\left[\begin{array}{c}
    \mathbf{h}^T\left(\mathbf{x}^*\right) \\
    \mathbf{H}
    \end{array}\right] \boldsymbol{\beta}, \sigma^{2}\left[\begin{array}{cc}
    c\left(\mathbf{x}^*, \mathbf{x}^*\right) & \mathbf{r}^T\left(\mathbf{x}^*\right) \\
    \mathbf{r}\left(\mathbf{x}^*\right) & \mathbf{R}
    \end{array}\right]\right)
    :label: jointystarytr

where :math:`\mathbf{r}(\mathbf{x}^*)=\left(c(\mathbf{x}^*, \mathbf{x}_{1}), \ldots, c(\mathbf{x}^*, \mathbf{x}_{n^{tr}}) \right)^T`.
According to the property of the multivariate Gaussian distribution, the conditional
distribution of :math:`y^*` conditioned on :math:`\mathbf{y}^{tr}` is again a
Gaussian distribution, namely

.. math:: p(y^* | \mathbf{y}^{tr}, \boldsymbol{\beta}, \sigma^{2}, \boldsymbol{\psi}) \sim  \mathcal{N}
    \left(m', \sigma^2 c' \right),\label{eq:conditionalGaussian}
    :label: conditionalGaussian

.. math:: m'=\mathbf{h}^T(\mathbf{x}^*) \boldsymbol{\beta} + \mathbf{r}^T(\mathbf{x}^*) \mathbf{R}^{-1} (\mathbf{y}^{tr}-\mathbf{H}\boldsymbol{\beta})
    :label: conditionalGaussianM

.. math:: c'=c(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{r}^T(\mathbf{x}^*) \mathbf{R}^{-1} \mathbf{r}(\mathbf{x}^*).
    :label: conditionalGaussianC

Equations :eq:`conditionalGaussian` to :eq:`conditionalGaussianC` can be easily
extended to any dimensionality since the joint distribution of the function
outputs at any number of untried inputs and :math:`\mathbf{y}^{tr}` also follows
a multivariate Gaussian distribution. Equations :eq:`conditionalGaussian` to 
:eq:`conditionalGaussianC` therefore essentially define the posterior which is an
updated Gaussian process, given the unknown regression parameters
:math:`\boldsymbol{\beta}`, variance :math:`\sigma^2`, and correlation length
parameters :math:`\boldsymbol{\psi}`. 

For any new input :math:`\mathbf{x}^*`, the output :math:`y^*` can be almost
instantaneously predicted, given by the mean :math:`m'`, namely equation
:eq:`conditionalGaussianM`. Moreover, the uncertainty associated with the prediction
can be estimated by for example the variance :math:`\sigma^2c'`. Depending on how
the unknown parameters (:math:`\boldsymbol{\beta}`, :math:`\sigma^2`, and :math:`\boldsymbol{\psi}`)
are learned based on the training data, the posterior can have various forms, see
`Section 3.3` of :cite:t:`Zhao2021b`.

Sample from GP
^^^^^^^^^^^^^^
A Gaussian process, either the prior or the posterior, defines a distribution over
functions. Once its mean function :math:`m(\cdot)` and covariance function
:math:`C(\cdot,\cdot)` are specified, the Gaussian process can be evaluated at a
finite number :math:`s` of input points :math:`\{\mathbf{x}_i\}_{i=1}^{s}`.
The joint distribution of :math:`\{y_i=f(\mathbf{x}_i)\}_{i=1}^{s}` follows a
:math:`s`-variate Gaussian distribution. A sample of the :math:`s`-variate Gaussian
distribution, denoted as :math:`\tilde{\mathbf{y}}:=(\tilde{y}_1,\ldots,\tilde{y}_s)^T`,
can be viewed as a sample of the Gaussian process (evaluated at a finite number of points).
To generate :math:`\tilde{\mathbf{y}}`, the :math:`s \times s` covariance matrix
:math:`\boldsymbol{\Sigma}` needs to be decomposed into the product of a lower triangular
matrix :math:`\mathbf{L}` and its conjugate transpose :math:`\mathbf{L}^T` using the Cholesky
decomposition

.. math:: \boldsymbol{\Sigma}=\left(\begin{array}{cccc}
    C\left(\mathbf{x}_{1}, \mathbf{x}_{1}\right) & C\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right) & \ldots & C\left(\mathbf{x}_{1}, \mathbf{x}_{s}\right) \\
    C\left(\mathbf{x}_{2}, \mathbf{x}_{1}\right) & C\left(\mathbf{x}_{2}, \mathbf{x}_{2}\right) & \ldots & C\left(\mathbf{x}_{2}, \mathbf{x}_{s}\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    C\left(\mathbf{x}_{s}, \mathbf{x}_{1}\right) & \ldots & \ldots & C\left(\mathbf{x}_{s}, \mathbf{x}_{s}\right)
    \end{array}\right)=\mathbf{L}\mathbf{L}^T.
    :label: Choleskydecomposition

Then a sample :math:`\tilde{\mathbf{y}}` can be obtained as follows:

.. math:: \tilde{\mathbf{y}} = (m(\mathbf{x}_i),\ldots,m(\mathbf{x}_s))^T + \mathbf{L}\mathbf{w},
    :label: GPsample

where :math:`\mathbf{w}:=(w_1,\dots,w_s)^T` denotes a :math:`s`-dimensional random vector
consisting of :math:`s` independent standard normal random variables :math:`w_i,i \in \{1,\ldots,s\}`.


ScalarGaSP Class
----------------

Class :class:`.ScalarGaSP` provides Gaussian process emulation for simulators 
which return a scalar value as output, namely :math:`y \in \mathbb{R}`. Please
refer to :cite:t:`Gu2018` for the detailed theory.

It is imported by::
    
    from psimpy.emulator.robustgasp import ScalarGaSP

Methods
^^^^^^^
.. autoclass:: psimpy.emulator.robustgasp.ScalarGaSP
    :members: train, predict, sample, loo_validate


PPGaSP Class
------------

Class :class:`.PPGaSP` provides Gaussian process emulation for simulators 
which return multiple values as output, namely :math:`y \in \mathbb{R}^k, k>1`.
Please refer to :cite:t:`Gu2016` for the detailed theory.

It is imported by::
    
    from psimpy.emulator.robustgasp import PPGaSP

Methods
^^^^^^^
.. autoclass:: psimpy.emulator.robustgasp.PPGaSP
    :members: train, predict, sample
