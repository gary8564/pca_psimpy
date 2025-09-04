*********
Simulator
*********

Computer simulations are widely used to study real-world systems in many fields.
Such a simulation model, so-called `simulator`, essentially defines a mapping from
an input space :math:`\mathcal{X}` to an output space :math:`\mathcal{Y}`.
More specifically, given a simulator :math:`\mathbf{y}=\mathbf{f}(\mathbf{x})`,
it maps a :math:`p`-dimensional input :math:`\mathbf{x} \in \mathcal{X} \subset{\mathbb{R}^p}`
to a :math:`k`-dimensional output :math:`\mathbf{y} \in \mathcal{Y} \subset{\mathbb{R}^k}`.
The mapping :math:`\mathbf{f}` can vary from simple linear equations which can be analytically
solved to complex partial differential equations which requires numerical schemes such as
finite element methods.

This module hosts simulators and functionality for running simulators. Currently
implemented classes are:

* :class:`.RunSimulator`: Serial and parallel execution of simulators.

* :class:`.MassPointModel`: Mass point model for landslide run-out simulation.

* :class:`.Ravaflow3GMixture`: Voellmy-type shallow flow model for landslide run-out simulation.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Simulator

    Run Simulator <run_simulator>
    Mass Point Model <mass_point_model>
    Ravaflow Mixture Model <ravaflow3G>

.. note::
   :class:`.MassPointModel` and :class:`.Ravaflow3GMixture` are only relevant if
   the user wants to perform run-out simulation. :class:`.MassPointModel` is purely
   Python-based and can be used right away. :class:`.Ravaflow3GMixture` depends on 
   `r.avaflow 3G <https://www.landslidemodels.org/r.avaflow/>`_, which needs to be
   installed by the user.