import os
import numpy as np
import pytest
from psimpy.simulator import MassPointModel

@pytest.mark.parametrize(
    "elevation, x0, y0",
    [
        (os.path.abspath(os.path.join(__file__, '../data/topo.asc')), 10, 10), 
        (os.path.abspath(os.path.join(__file__, '../data/synthetic_topo.asc')),
        -10, 10),
        (os.path.abspath(os.path.join(__file__, '../data/synthetic_topo.asc')),
        10, -10)
    ]
)
def test_preprocess_ValueError(elevation, x0, y0):
    mpm = MassPointModel()
    with pytest.raises(ValueError):
        mpm._preprocess(elevation, x0, y0)


@pytest.mark.parametrize(
    "coulomb_friction, turbulent_friction, x0, y0, ux0, uy0, dt, \
    tend, t0, g, atol, rtol, curvature",
    [
        (0.2, 800, 400, 2000, 0, 0, 2, 100, 0, 9.8, 1e-6, 1e-6, False), 
        (0.15, 1000, 500, 2000, 1, 1, 2, 200, 10, 9.8, 1e-5, 1e-5, False),
        (0.15, 1000, 500, 2000, 1, 1, 2, 200, 0, 10, 1e-5, 1e-5, True)
    ]
)
def test_mass_point_model_run(coulomb_friction, turbulent_friction, x0, y0, ux0,
    uy0, dt, tend, t0, g, atol, rtol, curvature):
    mpm = MassPointModel()
    elevation = os.path.abspath(os.path.join(__file__,
        '../data/synthetic_topo.asc'))
    output = mpm.run(
        elevation, coulomb_friction, turbulent_friction, x0, y0, ux0, uy0, dt,
        tend, t0, g, atol, rtol, curvature)
    assert isinstance(output, np.ndarray)
    assert len(output) > 0
    

