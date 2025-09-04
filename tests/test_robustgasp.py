import os
import pytest
import numpy as np
from rpy2.rinterface import NA
from beartype.roar import BeartypeCallHintParamViolation
from psimpy.emulator import ScalarGaSP, PPGaSP
from psimpy.emulator.robustgasp import RobustGaSPBase

@pytest.mark.parametrize(
    "ndim, zero_mean, nugget, nugget_est, range_par, method, prior_choice, \
    a, b, kernel_type, isotropic, optimization, alpha, lower_bound, max_eval, \
    num_initial_values",
    [
        (2.0, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, False, 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 1, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, 'No', NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, [1, 2], 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, None, 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', None, 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 1, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, 2,
        'matern_5_2', False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        None, False, 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', 'No', 'lbfgs', None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, None, None, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', 1.9, True, None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, 'Yes', None, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, 30.0, 2),
        (2, 'No', 0, False, NA, 'post_mode', 'ref_approx', 0.2, None,
        'matern_5_2', False, 'lbfgs', None, True, None, 2.0)
    ]
)
class Test_init_TypeError:

    def test_RobustGaSPbase_init_TypeError(self,
        ndim, zero_mean, nugget, nugget_est, range_par, method, prior_choice, a,
        b, kernel_type, isotropic, optimization, alpha, lower_bound, max_eval,
        num_initial_values):
        with pytest.raises(BeartypeCallHintParamViolation):
            RobustGaSPBase(
                ndim, zero_mean, nugget, nugget_est, range_par, method,
                prior_choice, a, b, kernel_type, isotropic, optimization, alpha,
                lower_bound, max_eval, num_initial_values)
    
    def test_ScalarGaSP_init_TypeError(self,
        ndim, zero_mean, nugget, nugget_est, range_par, method, prior_choice, a,
        b, kernel_type, isotropic, optimization, alpha, lower_bound, max_eval,
        num_initial_values):
        with pytest.raises(BeartypeCallHintParamViolation):
            ScalarGaSP(
                ndim, zero_mean, nugget, nugget_est, range_par, method,
                prior_choice, a, b, kernel_type, isotropic, optimization, alpha,
                lower_bound, max_eval, num_initial_values)
    
    def test_PPGaSP_init_TypeError(self,
        ndim, zero_mean, nugget, nugget_est, range_par, method, prior_choice, a,
        b, kernel_type, isotropic, optimization, alpha, lower_bound, max_eval,
        num_initial_values):
        with pytest.raises(BeartypeCallHintParamViolation):
            PPGaSP(
                ndim, zero_mean, nugget, nugget_est, range_par, method,
                prior_choice, a, b, kernel_type, isotropic, optimization, alpha,
                lower_bound, max_eval, num_initial_values)


@pytest.mark.parametrize(
    "design, response, trend",
    [
        (np.array([[[1,2]]]), np.array([0.1, 0.2]), None),
        (np.array([[1,1],[2,2],[3,3]]), np.array([0.1,0.2,0.3]), None),
        (np.array([1,2]), np.array([[[0.1,0.2]]]), None),
        (np.array([1,2]), np.array([0.1,0.2,0.3]), None),
        (np.array([1,2]), np.array([0.1,0.2]), np.array([[[1,1]]])),
        (np.array([1,2]), np.array([0.1,0.2]), np.array([1,1,1]))
    ]
)
def test_preprocess_design_response_trend(design, response, trend):
    rgasp_base = RobustGaSPBase(ndim=1)
    with pytest.raises(ValueError):
        rgasp_base._preprocess_design_response_trend(design, response, trend)


@pytest.mark.parametrize(
    "testing_input, testing_trend",
    [
        (np.array([1,2]), None),
        (np.array([[1,1],[2,2]]), np.array([1,1,1])),
        (np.array([[1,1],[2,2]]), np.ones((2,2)))
    ]
)
def test_preprocess_testing_input_trend(testing_input, testing_trend):
    rgasp_base = RobustGaSPBase(ndim=2)
    rgasp_base._preprocess_design_response_trend(
        design=np.array([[1,1],[2,2]]), response=np.array([0.1,0.2]), trend=None)
    with pytest.raises(ValueError):
        rgasp_base._preprocess_testing_input_trend(testing_input, testing_trend)


@pytest.mark.parametrize(
    "design, response,  nugget_est, method, kernel_type, testing_input",
    [
        (np.random.rand(5), np.random.rand(5), False, 'post_mode',
        'matern_5_2', np.random.rand(10)),
        (np.random.rand(5,1), np.random.rand(5), True, 'post_mode',
        'matern_5_2', np.random.rand(10)),
        (np.random.rand(5,3), np.random.rand(5,1), True, 'post_mode',
        'matern_5_2', np.random.rand(8,3)),
        (np.random.rand(5,3), np.random.rand(5,1), False, 'mmle',
        'matern_3_2', np.random.rand(10,3)),
        (np.random.rand(5,3), np.random.rand(5,1), False, 'mle',
        'pow_exp', np.random.rand(6,3))
    ]
)
def test_ScalarGaSP(design, response, nugget_est, method, kernel_type,
    testing_input):
    if design.ndim == 1:
        ndim = 1
    elif design.ndim == 2:
        ndim = design.shape[1]
    scalar_gasp = ScalarGaSP(ndim=ndim, nugget_est=nugget_est, method=method,
        kernel_type=kernel_type)

    assert scalar_gasp.emulator is None
    scalar_gasp.train(design, response)
    assert scalar_gasp.emulator is not None

    predictions = scalar_gasp.predict(testing_input)
    assert predictions.shape == (len(testing_input), 4)

    samples = scalar_gasp.sample(testing_input, nsamples=10)
    assert samples.shape == (len(testing_input), 10)

    samples = scalar_gasp.sample(testing_input, nsamples=1)
    assert samples.shape == (len(testing_input), 1)

    validation = scalar_gasp.loo_validate()
    assert validation.shape == (scalar_gasp.design.shape[0], 2)

dir_test = os.path.abspath(os.path.join(__file__, '../'))
humanityX = np.genfromtxt(os.path.join(dir_test,'data/humanityX.csv'),
    delimiter=',')
humanityY = np.genfromtxt(os.path.join(dir_test,'data/humanityY.csv'),
    delimiter=',')
humanityXt = np.genfromtxt(os.path.join(dir_test,'data/humanityXt.csv'),
    delimiter=',')
rgasp_predict = np.genfromtxt(os.path.join(dir_test,
    'data/humanity_rgasp_predict.csv'), delimiter=',')

def test_ScalarGaSP_vs_robustgasp():

    ndim = humanityX.shape[1]
    scalar_gasp = ScalarGaSP(ndim=ndim)
    scalar_gasp.train(humanityX, humanityY[:,0])
    scalar_gasp_predict = scalar_gasp.predict(humanityXt)

    assert scalar_gasp_predict.shape == rgasp_predict.shape
    assert np.all(
        np.abs((scalar_gasp_predict - rgasp_predict) / rgasp_predict) < 0.01
        )
    assert np.mean(
        np.abs((scalar_gasp_predict - rgasp_predict) / rgasp_predict)) < 0.001


@pytest.mark.parametrize(
    "design, response,  nugget_est, method, kernel_type, testing_input",
    [
        (np.random.rand(5), np.random.rand(5,2), False, 'post_mode',
        'matern_5_2', np.random.rand(10)),
        (np.random.rand(5,1), np.random.rand(5,10), True, 'post_mode',
        'matern_5_2', np.random.rand(10)),
        (np.random.rand(5,3), np.random.rand(5,10), True, 'post_mode',
        'matern_5_2', np.random.rand(8,3)),
        (np.random.rand(5,3), np.random.rand(5,10), False, 'mmle',
        'matern_3_2', np.random.rand(10,3)),
        (np.random.rand(5,3), np.random.rand(5,10), False, 'mle',
        'pow_exp', np.random.rand(6,3))
    ]
)
def test_PPGaSP(design, response, nugget_est, method, kernel_type,
    testing_input):
    if design.ndim == 1:
        ndim = 1
    elif design.ndim == 2:
        ndim = design.shape[1]
    
    ppgasp = PPGaSP(ndim=ndim, nugget_est=nugget_est, method=method,
        kernel_type=kernel_type)
    
    assert ppgasp.emulator is None
    ppgasp.train(design, response)
    assert ppgasp.emulator is not None

    predictions = ppgasp.predict(testing_input)
    assert predictions.shape == (len(testing_input), response.shape[1], 4)

    samples = ppgasp.sample(testing_input, nsamples=10)
    assert samples.shape == (len(testing_input), response.shape[1], 10)

    samples = ppgasp.sample(testing_input, nsamples=1)
    assert samples.shape == (len(testing_input), response.shape[1], 1)


expected_mean = np.genfromtxt(os.path.join(dir_test,
    'data/humanity_ppgasp_predict_mean.csv'), delimiter=',')
expected_lower95 = np.genfromtxt(os.path.join(dir_test,
    'data/humanity_ppgasp_predict_lower95.csv'), delimiter=',')
expected_upper95 = np.genfromtxt(os.path.join(dir_test,
    'data/humanity_ppgasp_predict_upper95.csv'), delimiter=',')
expected_sd = np.genfromtxt(os.path.join(dir_test,
    'data/humanity_ppgasp_predict_sd.csv'), delimiter=',')
    
def test_PPGaSP_vs_robustgasp():
    ndim = humanityX.shape[1]
    ppgasp = PPGaSP(ndim=ndim, nugget_est=True)
    ppgasp.train(humanityX, humanityY)
    ppgasp_predict = ppgasp.predict(humanityXt)

    ppgasp_predict_mean = ppgasp_predict[:,:,0]
    ppgasp_predict_lower95 = ppgasp_predict[:,:,1]
    ppgasp_predict_upper95 = ppgasp_predict[:,:,2]
    ppgasp_predict_sd = ppgasp_predict[:,:,3]

    diff_mean = np.abs(ppgasp_predict_mean - expected_mean) / \
        (np.max(expected_mean, axis=0) - np.min(expected_mean, axis=0))
    assert np.all(diff_mean < 0.01)
    assert np.mean(diff_mean) < 0.001

    diff_lower95 = np.abs(ppgasp_predict_lower95 - expected_lower95) / \
        (np.max(expected_lower95, axis=0) - np.min(expected_lower95, axis=0))
    assert np.all(diff_lower95 < 0.01)
    assert np.mean(diff_lower95) < 0.001

    diff_upper95 = np.abs(ppgasp_predict_upper95 - expected_upper95) / \
        (np.max(expected_upper95, axis=0) - np.min(expected_upper95, axis=0))
    assert np.all(diff_upper95 < 0.01)
    assert np.mean(diff_upper95) < 0.001

    diff_sd = np.abs(ppgasp_predict_sd - expected_sd) / \
        (np.max(expected_sd, axis=0) - np.min(expected_sd, axis=0))
    assert np.all(diff_sd < 0.01)
    assert np.mean(diff_sd) < 0.002

    
