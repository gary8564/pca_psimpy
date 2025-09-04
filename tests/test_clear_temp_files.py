import os
import shutil
import pytest

@pytest.mark.order(-1)
def test_clear_temp_files():

    dir_test = os.path.abspath(os.path.join(__file__, '../'))

    temp_dirs = [
        os.path.join(dir_test, 'temp_add'), # test_run_simulator
        os.path.join(dir_test, 'temp_run_add'), # test_run_simulator
        os.path.join(dir_test, 'temp_run_mass_point_model'), # test_run_mass_point_model
        os.path.join(dir_test, 'temp_ravaflow_1'), # test_ravaflow24
        os.path.join(dir_test, 'temp_ravaflow_2'), # test_ravaflow24
        os.path.join(dir_test, 'temp_run_ravaflow'), # test_run_ravaflow24
        os.path.join(dir_test, 'temp_ravaflow_results'), # test_run_ravaflow24
        os.path.join(dir_test, 'temp_metropolis_hastings'), # test_metropolis_hasting
        os.path.join(dir_test, 'temp_bayes_inference'), # test_bayes_inference
        os.path.join(dir_test, 'temp_active_learning') # test_active_learning
    ]

    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)