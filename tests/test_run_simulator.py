import os
import pytest
import numpy as np
from beartype.roar import BeartypeCallHintParamViolation
from psimpy.simulator import RunSimulator

def add(a, b, c , d=100, save=False, filename=None):
    if save is True:
        test_folder = os.path.abspath(os.path.join(__file__,'../'))
        os.chdir(test_folder)
        if not os.path.exists("temp_add"):
            os.mkdir("temp_add")
        fname = f"{filename}.txt"
        np.savetxt(os.path.join(test_folder, "temp_add", fname),
            np.array([a,b,c,d]))
    return a + b + c + d


@pytest.mark.parametrize(
    "simulator, var_inp_parameter, fix_inp, o_parameter, dir_out, save_out",
    [
        (add, ("a", "b", "c"), {"save":False, "d":100}, None, None, False), 
        (add, ["a", "b", "c"], ["save", "d"], None, None, False),
        (add, ["a", "b", "c"], {"save":False, "d":100}, ["filename"], None,
         False),
        (add, ["a", "b", "c"], {"save":False, "d":100}, "filename", True,
         False),
        (add, ["a", "b", "c"], {"save":False, "d":100}, "filename", None,
         "Yes")
    ]
)
def test_RunSimulator_init_TypeError(simulator, var_inp_parameter, fix_inp,
    o_parameter, dir_out, save_out):
    with pytest.raises(BeartypeCallHintParamViolation):
        RunSimulator(simulator, var_inp_parameter, fix_inp, o_parameter,
            dir_out, save_out)


@pytest.mark.parametrize(
    "dir_out, var_samples, prefixes, save_out, append",
    [
        (None, np.random.rand(5,3), None, True, False),
        (f"{os.path.abspath(os.path.join(__file__,'../sim_log'))}", 
         np.random.rand(5,3), None, True, False),
        (f"{os.path.abspath(os.path.join(__file__,'../data/mpm_topo.asc'))}", 
         np.random.rand(5,3), None, True, False),
        (None, np.random.rand(5,4), None, False, False),
        (None, np.random.rand(5,3), None, False, True),
        (None, np.random.rand(5,3), ['sim0', 'sim1'], False, False),
        (None, np.random.rand(5,3), ['sim0', 'sim1', 'sim2', 'sim3', 'sim3'],
        False, False)
    ]
)
def test_RunSimulator_ValueError(
    dir_out, var_samples, prefixes, save_out, append):
    with pytest.raises(ValueError):
        run_add = RunSimulator(simulator=add, var_inp_parameter=['a','b','c'],
            dir_out=dir_out, save_out=save_out)
        _ = run_add._preprocess(var_samples, prefixes, append)


def test_RunSimulator_serial_parallel_run():
    run_add = RunSimulator(simulator=add, var_inp_parameter=['a','b','c'],
        fix_inp={'d':100, 'save':False}, o_parameter=None, dir_out=None,
        save_out=False)
    
    run_add.serial_run(var_samples=np.array([[1,2,3],[10,20,30]]), append=False)
    assert run_add.outputs == [106, 160]

    run_add.parallel_run(var_samples=np.array([[1,2,3],[10,20,30]]), append=False)
    assert run_add.outputs == [106, 160]

    run_add.serial_run(var_samples=np.array([[5,5,5],[10,10,10]]), append=True)
    assert run_add.outputs == [106, 160, 115, 130]

    run_add.parallel_run(var_samples=np.array([[15,15,15],[1,1,1],[2,2,2],
        [3,3,3]]), append=True, max_workers=4)
    assert run_add.outputs == [106, 160, 115, 130, 145, 103, 106, 109]


def test_RunSimulator_serial_parallel_run_with_o_parameter():
    run_add = RunSimulator(simulator=add, var_inp_parameter=['a','b','c'],
        fix_inp={'d':100, 'save':True}, o_parameter='filename', dir_out=None,
        save_out=False)
    
    run_add.serial_run(var_samples=np.array([[1,2,3],[10,20,30]]),
        prefixes=['serial_run0', 'serial_run1'], append=False)
    assert run_add.outputs == [106, 160]
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../')),
        'temp_add', 'serial_run0.txt')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../')),
        'temp_add', 'serial_run1.txt')
    )

    run_add.parallel_run(var_samples=np.array([[10,10,10],[100,100,100]]),
        prefixes=['parallel_run0', 'parallel_run1'], append=True)
    assert run_add.outputs == [106, 160, 130, 400]
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../')),
        'temp_add', 'parallel_run0.txt')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../')),
        'temp_add', 'parallel_run1.txt')
    )


def test_RunSimulator_serial_parallel_run_with_save_out():
    dir_out = os.path.join(os.path.abspath(os.path.join(__file__,'../')),
        'temp_run_add')
    os.mkdir(dir_out)
    run_add = RunSimulator(simulator=add, var_inp_parameter=['a','b','c'],
        fix_inp={'d':100, 'save':True}, o_parameter='filename', dir_out=dir_out,
        save_out=True)
    
    run_add.serial_run(var_samples=np.array([[1,2,3],[10,20,30]]),
        prefixes=['serial_run0', 'serial_run1'], append=False)
    assert run_add.outputs == [106, 160]
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_add')),
        'serial_run0.txt')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_run_add')),
        'serial_run0_output.npy')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_add')),
        'serial_run1.txt')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_run_add')),
        'serial_run1_output.npy')
    )

    run_add.parallel_run(var_samples=np.array([[10,10,10],[100,100,100]]),
        prefixes=['parallel_run0', 'parallel_run1'], append=True)
    assert run_add.outputs == [106, 160, 130, 400]
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_add')),
        'parallel_run0.txt')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_run_add')),
        'parallel_run0_output.npy')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_add')),
        'parallel_run1.txt')
    )
    assert os.path.exists(
        os.path.join(os.path.abspath(os.path.join(__file__,'../temp_run_add')),
        'parallel_run1_output.npy')
    )



