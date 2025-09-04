import os
import pytest
import numpy as np
from psimpy.simulator import Ravaflow3GMixture


@pytest.mark.parametrize(
    "dir_sim, diffusion_control, curvature_control, surface_control, \
    entrainment_control, stopping_control",
    [
        (
            os.path.abspath(os.path.join(__file__, "../not_exist_dir")),
            "0",
            "0",
            "0",
            "0",
            "0",
        ),
        (
            os.path.abspath(os.path.join(__file__, "../data/synthetic_rel.tif")),
            "0",
            "0",
            "0",
            "0",
            "0",
        ),
        (os.path.abspath(os.path.join(__file__, "../data")), "2", "0", "0", "0", "0"),
        (os.path.abspath(os.path.join(__file__, "../data")), "0", "3", "0", "0", "0"),
        (os.path.abspath(os.path.join(__file__, "../data")), "0", "0", "2", "0", "0"),
        (os.path.abspath(os.path.join(__file__, "../data")), "0", "0", "0", "5", "0"),
        (os.path.abspath(os.path.join(__file__, "../data")), "0", "0", "0", "0", "4"),
    ],
)
def test_ravaflow3G_mixture_init_ValueError(
    dir_sim,
    diffusion_control,
    curvature_control,
    surface_control,
    entrainment_control,
    stopping_control,
):
    with pytest.raises(ValueError):
        _ = Ravaflow3GMixture(
            dir_sim=dir_sim,
            diffusion_control=diffusion_control,
            curvature_control=curvature_control,
            surface_control=surface_control,
            entrainment_control=entrainment_control,
            stopping_control=stopping_control,
        )


@pytest.mark.parametrize(
    "prefix, elevation, hrelease",
    [
        (
            "sim",
            os.path.abspath(os.path.join(__file__, "../data/synthetic_topo.tif")),
            os.path.abspath(os.path.join(__file__, "../data/synthetic_rel.tif")),
        ),
        (
            "sim_new",
            os.path.abspath(os.path.join(__file__, "../data/syn_topo.tif")),
            os.path.abspath(os.path.join(__file__, "../data/synthetic_rel.tif")),
        ),
        (
            "sim_new",
            os.path.abspath(os.path.join(__file__, "../data/synthetic_topo.tif")),
            os.path.abspath(os.path.join(__file__, "../data/syn_rel.tif")),
        ),
    ],
)
def test_ravaflow3G_mixture_preprocess_ValueError(prefix, elevation, hrelease):
    dir_test = os.path.abspath(os.path.join(__file__, "../"))
    os.chdir(dir_test)
    if not os.path.exists("temp_ravaflow_1"):
        os.mkdir("temp_ravaflow_1")
    os.chdir("temp_ravaflow_1")
    dir_sim = os.path.join(dir_test, "temp_ravaflow_1")
    if not os.path.exists("sim_results"):
        os.mkdir("sim_results")

    ravaflow3G_mixture = Ravaflow3GMixture(dir_sim=dir_sim)
    with pytest.raises(ValueError):
        ravaflow3G_mixture.preprocess(
            prefix=prefix, elevation=elevation, hrelease=hrelease
        )


def test_ravaflow3G_mixture_run_and_extract_output():
    dir_test = os.path.abspath(os.path.join(__file__, "../"))
    os.chdir(dir_test)
    if not os.path.exists("temp_ravaflow_2"):
        os.mkdir("temp_ravaflow_2")
    dir_sim = os.path.join(dir_test, "temp_ravaflow_2")

    ravaflow3G_mixture = Ravaflow3GMixture(dir_sim=dir_sim, time_end=100)

    elevation = os.path.join(dir_test, "data/synthetic_topo.tif")
    hrelease = os.path.join(dir_test, "data/synthetic_rel.tif")
    prefix = "test"

    grass_location, sh_file = ravaflow3G_mixture.preprocess(
        prefix=prefix, elevation=elevation, hrelease=hrelease, EPSG="2326"
    )

    ravaflow3G_mixture.run(grass_location, sh_file)

    assert os.path.exists(os.path.join(dir_sim, f"{prefix}_results"))

    impact_area = ravaflow3G_mixture.extract_impact_area(prefix)
    print(f"impact_area: {impact_area}")
    assert isinstance(impact_area, float)
    assert impact_area > 0

    overall_max_velocity = ravaflow3G_mixture.extract_qoi_max(prefix, "v")
    print(f"overall_max_velocity: {overall_max_velocity}")
    assert isinstance(overall_max_velocity, float)
    assert overall_max_velocity > 0

    raster_max_velocity = ravaflow3G_mixture.extract_qoi_max(
        prefix, "v", aggregate=False
    )
    assert isinstance(raster_max_velocity, np.ndarray)
    assert np.max(raster_max_velocity) == overall_max_velocity

    loc = np.array([[500, 2000], [600, 2000], [700, 2500]])
    loc_max_pressure = ravaflow3G_mixture.extract_qoi_max_loc(prefix, loc, "p")
    print(f"loc_max_pressure: {loc_max_pressure}")
    assert isinstance(loc_max_pressure, np.ndarray)
    assert loc_max_pressure.ndim == 1
    assert len(loc_max_pressure) == len(loc)

    loc = np.array([[500, 2000]])
    loc_max_energy = ravaflow3G_mixture.extract_qoi_max_loc(prefix, loc, "t")
    print(f"loc_max_energy: {loc_max_energy}")
    assert isinstance(loc_max_energy, np.ndarray)
    assert loc_max_energy.ndim == 1
    assert len(loc_max_energy) == len(loc)
