********
Emulator
********

This module hosts functionality for emulation methods.

Emulation, also known as surrogate modeling or meta-modeling, is a type of
method which is used to build cheap-to-evaluate emulators (surrogate models) to
approximate expensive-to-evaluate simulators. It can greatly reduce
computational costs of tasks in which a computationally intensive simulator
needs to be run multiple times, such as a global sensitivity anlaysis,
uncertainty quantification, or parameter calibration.

Currently implemented classes in this module are:

* :class:`.ScalarGaSP`: Gaussian process emulation for simulators which return a
  scalar value as output.

* :class:`.PPGaSP`: Gaussian process emulation for simulators which return
  multiple values as output. 

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Emulator

    RobustGaSP <robustgasp>