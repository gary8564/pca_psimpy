import os
import time
import itertools
import numpy as np
from psimpy.simulator import RunSimulator
from psimpy.simulator import Ravaflow3GMixture

dir_test = os.path.abspath(os.path.join(__file__, "../"))

os.chdir(dir_test)
if not os.path.exists("temp_run_ravaflow"):
    os.mkdir("temp_run_ravaflow")
dir_out = os.path.join(dir_test, "temp_run_ravaflow")

if not os.path.exists("temp_ravaflow_results"):
    os.mkdir("temp_ravaflow_results")
dir_sim = os.path.join(dir_test, "temp_ravaflow_results")

elevation = os.path.join(dir_test, "data/synthetic_topo.tif")
hrelease = os.path.join(dir_test, "data/synthetic_rel.tif")
loc = np.array([[500, 2000], [600, 2000], [700, 2500]])

ravaflow3G_mixture = Ravaflow3GMixture(dir_sim=dir_sim, time_end=100)


# define simulator
def simulator(
    prefix, elevation, hrelease, basal_friction, turbulent_friction, EPSG, qoi, loc
):
    grass_location, sh_file = ravaflow3G_mixture.preprocess(
        prefix=prefix,
        elevation=elevation,
        hrelease=hrelease,
        basal_friction=basal_friction,
        turbulent_friction=turbulent_friction,
        EPSG=EPSG,
    )

    ravaflow3G_mixture.run(grass_location, sh_file)

    impact_area = ravaflow3G_mixture.extract_impact_area(prefix)
    overall_max_qoi = ravaflow3G_mixture.extract_qoi_max(prefix, qoi)
    loc_max_qoi = ravaflow3G_mixture.extract_qoi_max_loc(prefix, loc, qoi)

    output = np.zeros(len(loc_max_qoi) + 2)
    output[0] = impact_area
    output[1] = overall_max_qoi
    output[2:] = loc_max_qoi

    return output


def test_run_ravaflow3G():
    var_inp_parameter = ["basal_friction", "turbulent_friction"]
    fix_inp = {
        "elevation": elevation,
        "hrelease": hrelease,
        "qoi": "v",
        "loc": loc,
        "EPSG": "2326",
    }
    o_parameter = "prefix"

    run_ravaflow3G = RunSimulator(
        simulator=simulator,
        var_inp_parameter=var_inp_parameter,
        fix_inp=fix_inp,
        o_parameter=o_parameter,
        dir_out=dir_out,
        save_out=True,
    )

    basal_friction = [20]
    turbulent_friction = [3, 4]
    var_samples = np.array(
        [x for x in itertools.product(basal_friction, turbulent_friction)]
    )

    serial_prefixes = ["serial" + str(i) for i in range(len(var_samples))]

    start = time.time()
    run_ravaflow3G.serial_run(var_samples=var_samples, prefixes=serial_prefixes)
    serial_time = time.time() - start
    serial_output = run_ravaflow3G.outputs
    print(f"serial_output: {serial_output}")
    assert len(serial_output) == len(var_samples)

    for i in range(len(var_samples)):
        assert isinstance(serial_output[i], np.ndarray)
        assert len(serial_output[i]) == len(loc) + 2

    parallel_prefixes = ["parallel" + str(i) for i in range(len(var_samples))]

    start = time.time()
    run_ravaflow3G.parallel_run(
        var_samples, prefixes=parallel_prefixes, max_workers=2, append=True
    )
    parallel_time = time.time() - start

    assert len(run_ravaflow3G.outputs) == 2 * len(var_samples)
    parallel_output = run_ravaflow3G.outputs[len(var_samples) :]
    print(f"parallel_output: {parallel_output}")
    assert len(parallel_output) == len(var_samples)

    print("Serial run time: ", serial_time)
    print("Parallel run time: ", parallel_time)
    assert serial_time > parallel_time

    for i in range(len(var_samples)):
        assert np.array_equal(serial_output[i], parallel_output[i])
