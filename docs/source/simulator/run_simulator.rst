Run Simulator
=============

A `simulator` which describes certain real-world system is often subject to
uncertainties. Uncertainty-related analyses require running the `simulator`
multiple times at different variable input points. Class :class:`.RunSimulator`
implemented in this module provides functionality to sequentially or parallelly
execute `simulator` at a number of varaible input points.

The user needs to define their `simulator`, implement an interface (essentially
a Python function) to call the `simulator` from within Python and return output
of interest, and inform :class:`.RunSimulator` the signature of the interface.


RunSimulator Class
------------------

The :class:`.RunSimulator` class is imported by::
    
    from psimpy.simulator.run_simulator import RunSimulator

Methods
^^^^^^^
.. autoclass:: psimpy.simulator.run_simulator.RunSimulator
    :members: serial_run, parallel_run

Attributes
^^^^^^^^^^
.. autoattribute:: psimpy.simulator.run_simulator.RunSimulator.var_samples
.. autoattribute:: psimpy.simulator.run_simulator.RunSimulator.outputs