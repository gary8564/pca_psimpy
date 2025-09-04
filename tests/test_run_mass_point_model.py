import os
import time
import itertools
import numpy as np
from psimpy.simulator import RunSimulator
from psimpy.simulator import MassPointModel

def test_run_mass_point_model():
    mpm = MassPointModel()
    model = mpm.run
    elev = os.path.abspath(os.path.join(__file__, '../data/synthetic_topo.asc'))
    var_inp_parameter = ['coulomb_friction', 'turbulent_friction']
    fix_inp = {'elevation': elev, 'x0': 400, 'y0': 2000, 'dt': 2, 'tend': 400}

    test_folder = os.path.abspath(os.path.join(__file__,'../'))
    os.chdir(test_folder)
    if not os.path.exists('temp_run_mass_point_model'):
        os.mkdir('temp_run_mass_point_model')
    dir_out = os.path.abspath(os.path.join(__file__,
        '../temp_run_mass_point_model'))

    run_model = RunSimulator(
        simulator=model, var_inp_parameter=var_inp_parameter,
        fix_inp=fix_inp, dir_out=dir_out, save_out=True)
    
    coulomb_friction = np.arange(0.1, 0.31, 0.2)
    turbulent_friction = np.arange(500, 2001, 400)
    var_samples = np.array(
        [x for x in itertools.product(coulomb_friction, turbulent_friction)])
    
    serial_prefixes = ["serial"+str(i) for i in range(len(var_samples))]
    parallel_prefixes = ["parallel"+str(i) for i in range(len(var_samples))]

    start = time.time()
    run_model.serial_run(var_samples=var_samples, prefixes=serial_prefixes)
    serial_time = time.time() - start
    serial_output = run_model.outputs
    
    start = time.time()
    run_model.parallel_run(var_samples, prefixes=parallel_prefixes,
        max_workers=4)
    parallel_time = time.time() - start
    parallel_output = run_model.outputs
    
    print("Serial run time: ", serial_time)
    print("Parallel run time: ", parallel_time)

    assert len(serial_output) == len(parallel_output)
    for i in range(len(serial_output)):
        assert np.max(np.abs(serial_output[i] - parallel_output[i])) < 1e-5
    
    assert serial_time > parallel_time
