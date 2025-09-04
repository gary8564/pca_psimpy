import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, multivariate_normal, uniform
from beartype.roar import BeartypeCallHintParamViolation
from psimpy.inference import ActiveLearning
from psimpy.simulator import RunSimulator
from psimpy.simulator import MassPointModel
from psimpy.sampler import LHS
from psimpy.sampler import Saltelli
from psimpy.emulator import ScalarGaSP, PPGaSP
from psimpy.inference import GridEstimation
from psimpy.inference import MetropolisHastingsEstimation
from psimpy.sampler import MetropolisHastings

dir_test = os.path.abspath(os.path.join(__file__, '../'))

@pytest.mark.parametrize(
    "run_sim_obj, prior, likelihood, lhs_sampler, scalar_gasp, optimizer",
    [
        (MassPointModel(), uniform.pdf, norm.pdf, LHS(1), ScalarGaSP(1),
        optimize.brute), 
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        None, norm.pdf, LHS(1), ScalarGaSP(1), optimize.brute),
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        uniform.pdf, None, LHS(1), ScalarGaSP(1), optimize.brute),
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        uniform.pdf, norm.pdf, Saltelli(1), ScalarGaSP(1), optimize.brute),
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        uniform.pdf, norm.pdf, LHS(1), PPGaSP(1), optimize.brute),
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        uniform.pdf, norm.pdf, LHS(1), ScalarGaSP(1), None)       
    ]
)  
def test_ActiveLearning_init_TypeError(run_sim_obj, prior, likelihood,
    lhs_sampler, scalar_gasp, optimizer):
    ndim = 1
    bounds = np.array([[0,1]])
    data  = np.array([1,2,3])
    with pytest.raises(BeartypeCallHintParamViolation):
        _ = ActiveLearning(ndim, bounds, data, run_sim_obj, prior, likelihood,
            lhs_sampler, scalar_gasp, optimizer=optimizer)


@pytest.mark.parametrize(
    "run_sim_obj, lhs_sampler, scalar_gasp",
    [
        (RunSimulator(MassPointModel.run,
            ['coulomb_friction', 'turbulent_friction']),
        LHS(1), ScalarGaSP(1)
        ),
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        LHS(2), ScalarGaSP(1)),
        (RunSimulator(MassPointModel.run, ['coulomb_friction']),
        LHS(1), ScalarGaSP(3))      
    ]
)  
def test_ActiveLearning_init_RuntimeError(run_sim_obj, lhs_sampler,
    scalar_gasp):
    ndim = 1
    bounds = np.array([[0,1]])
    data  = np.array([1,2,3])
    prior = uniform.pdf
    likelihood = norm.pdf
    with pytest.raises(RuntimeError):
        _ = ActiveLearning(ndim, bounds, data, run_sim_obj, prior, likelihood,
            lhs_sampler, scalar_gasp)


@pytest.mark.parametrize(
    "scalar_gasp_trend, indicator",
    [
        ('cubic', 'entropy'),
        ('linear', 'divergence')
    ]
)  
def test_ActiveLearning_init_NotImplementedError(scalar_gasp_trend, indicator):
    ndim = 1
    bounds = np.array([[0,1]])
    data  = np.array([1,2,3])
    run_sim_obj = RunSimulator(MassPointModel.run, ['coulomb_friction'])
    lhs_sampler = LHS(1)
    scalar_gasp = ScalarGaSP(1)
    prior = uniform.pdf
    likelihood = norm.pdf
    with pytest.raises(NotImplementedError):
        _ = ActiveLearning(ndim, bounds, data, run_sim_obj, prior, likelihood,
            lhs_sampler, scalar_gasp, scalar_gasp_trend=scalar_gasp_trend,
            indicator=indicator)

def test_ActiveLearning_init_ValueError():
    ndim = 1
    bounds = np.array([[0,1]])
    data  = np.array([1,2,3])
    run_sim_obj = RunSimulator(MassPointModel.run, ['coulomb_friction'])
    lhs_sampler = LHS(1)
    scalar_gasp = ScalarGaSP(1)
    prior = uniform.pdf
    likelihood = norm.pdf
    kwgs_optimizer = {"NS":50}
    with pytest.raises(ValueError):
        _ = ActiveLearning(ndim, bounds, data, run_sim_obj, prior, likelihood,
            lhs_sampler, scalar_gasp, kwgs_optimizer=kwgs_optimizer)


def create_temp_dirs():
    if not os.path.exists(os.path.join(dir_test, 'temp_active_learning')):
        os.chdir(dir_test)
        os.mkdir('temp_active_learning')
        os.chdir('temp_active_learning')
        os.mkdir('simulator_internal_outputs')
        os.mkdir('run_simulator_outputs')

def f(x1, x2, dir_sim, output_name):
    """Set simulator as y=x for Rosenbrock function."""
    np.savetxt(os.path.join(dir_sim, f'{output_name}.txt'), np.array([x1,x2]))
    return np.array([x1,x2])

def create_active_learner():

    def prior(x):
        """Uniform prior U[-5,5]x[-5,5]"""
        return 1/100

    def likelihood(y, data):
        """Rosenbrock function."""
        return np.exp(-(y[0]-data)**2/100 - (y[0]**2-y[1])**2)
    
    fix_inp = {'dir_sim': os.path.join(
        dir_test, 'temp_active_learning/simulator_internal_outputs')
        }
    run_Rosenbrock = RunSimulator(
        f,
        var_inp_parameter=['x1','x2'], 
        fix_inp=fix_inp,
        o_parameter = 'output_name',
        dir_out=os.path.join(
            dir_test, 'temp_active_learning/run_simulator_outputs'),
        save_out=True)
    ndim = 2
    bounds = np.array([[-5,5],[-5,5]])
    data = np.array([1])
    lhs_sampler = LHS(ndim=ndim, bounds=bounds)
    scalar_gasp = ScalarGaSP(ndim=ndim)
    
    return ActiveLearning(ndim, bounds, data, run_Rosenbrock, prior, likelihood,
        lhs_sampler, scalar_gasp)

Ns = 40
n0 = 40
niter = 60
nbins = [50, 50]

test_entropy_lhs_seed = 1
test_entropy_mh_seed = 1

test_variance_lhs_seed = 10
test_variance_mh_seed = 1

def test_ActiveLearning_brute_entropy():
    test_name = 'brute_entropy'
    create_temp_dirs()
    al_entropy = create_active_learner()
    al_entropy.optimizer = optimize.brute
    al_entropy.indicator = 'entropy'
    al_entropy.kwgs_optimizer['Ns'] = Ns
    al_entropy.lhs_sampler.rng = np.random.default_rng(test_entropy_lhs_seed)

    assert al_entropy.scalar_gasp.emulator is None

    prefixes = [f'init_{test_name}_sim{i}' for i in range(n0)]
    init_var_samples, init_sim_outputs = \
        al_entropy.initial_simulation(n0, prefixes, mode='parallel',
            max_workers=4)
    
    assert init_var_samples.shape == (n0, al_entropy.ndim)
    assert init_sim_outputs.shape[0] == n0
    
    iter_prefixes = [f'iter_{test_name}_sim{i}' for i in range(niter)]
    var_samples, _, _ = al_entropy.iterative_emulation(
        n0, init_var_samples, init_sim_outputs, niter=niter,
        iter_prefixes=iter_prefixes)
    
    txt_file = os.path.join(dir_test,
        f'temp_active_learning/{test_name}_var_samples.txt')
    np.savetxt(txt_file, var_samples, delimiter=',')
    
    # estimate posterior using grip estimation based on final GP emulator
    grid_estimator = GridEstimation(
        ndim=al_entropy.ndim,
        bounds=al_entropy.bounds,
        ln_pxl=al_entropy.approx_ln_pxl)
    
    posterior, x_ndim = grid_estimator.run(nbins)

    # estimate posterior using mh estimation based on final GP emulator
    mh_estimator = MetropolisHastingsEstimation(
        ndim=al_entropy.ndim,
        bounds=al_entropy.bounds,
        ln_pxl=al_entropy.approx_ln_pxl)

    init_state = np.array([-4,-4])
    f_sample = multivariate_normal.rvs
    nburn = 100
    nthin = 10
    kwgs_f_sample = {'random_state': np.random.default_rng(test_entropy_mh_seed)}

    mh_sampler = MetropolisHastings(
        ndim=al_entropy.ndim,
        init_state=init_state,
        f_sample=f_sample,
        bounds=al_entropy.bounds,
        nburn=nburn,
        nthin=nthin,
        seed=test_entropy_mh_seed,
        kwgs_f_sample=kwgs_f_sample)
    
    nsamples = 5000
    mh_samples, mh_accept = mh_estimator.run(nsamples, mh_sampler)

    assert mh_samples.shape == (nsamples, al_entropy.ndim)
    assert mh_accept.shape == (nsamples,)
    print(f'{test_name} mh acceptance ratio: ', np.mean(mh_accept))

    # plot
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(init_var_samples[:,0], init_var_samples[:,1], s=10, c='r',
        marker='o', zorder=1, alpha=0.8)
    axes[0].scatter(var_samples[n0:,0], var_samples[n0:,1], s=15, c='k',
        marker='+', zorder=2, alpha=0.8)
    
    posterior = np.where(posterior < 1e-10, np.nan, posterior)

    contour = axes[0].contour(x_ndim[0], x_ndim[1], np.transpose(posterior),
        levels=10, zorder=0)
    plt.colorbar(contour, ax=axes[0])
    
    axes[0].set_title(
        'Active learning \n prior U[-5,5]x[-5,5] \n simulator y=x \n'
        'likelihood Rosenbrock \n '+
        f'n0={n0}, actual_niter={len(var_samples)-n0}' + '\n' +
        f'{test_name} grid approximation')
    
    axes[1].scatter(mh_samples[:, 0],mh_samples[:,1], s=10, c='r',
        marker='o', alpha=0.1)
    axes[1].set_title(
        'Active learning \n prior U[-5,5]x[-5,5] \n simulator y=x \n'
        'likelihood Rosenbrock \n '+
        f'n0={n0}, actual_niter={len(var_samples)-n0}' + '\n' +
        f'{test_name} metropolis hastings')
    
    for i in range(2):
        axes[i].set_xlim(-5.1, 5.1)
        axes[i].set_ylim(-5.1, 5.1)
        axes[i].set_xlabel('x1')
        axes[i].set_ylabel('x2')

    png_file = os.path.join(dir_test,
        f'temp_active_learning/{test_name}.png')
    plt.savefig(png_file, bbox_inches='tight')

    assert os.path.exists(png_file)


def test_ActiveLearning_brute_variance():
    test_name = 'brute_variance'
    create_temp_dirs()
    al_variance = create_active_learner()
    al_variance.optimizer = optimize.brute
    al_variance.indicator = 'variance'
    al_variance.kwgs_optimizer['Ns'] = Ns
    al_variance.lhs_sampler.rng = np.random.default_rng(test_variance_lhs_seed)

    assert al_variance.scalar_gasp.emulator is None

    prefixes = [f'init_{test_name}_sim{i}' for i in range(n0)]
    init_var_samples, init_sim_outputs = \
        al_variance.initial_simulation(n0, prefixes, mode='serial')
    
    assert init_var_samples.shape == (n0, al_variance.ndim)
    assert init_sim_outputs.shape[0] == n0

    iter_prefixes = [f'iter_{test_name}_sim{i}' for i in range(niter)]
    var_samples, _, _ = al_variance.iterative_emulation(
        n0, init_var_samples, init_sim_outputs, niter=niter,
        iter_prefixes=iter_prefixes)
    
    txt_file = os.path.join(dir_test,
        f'temp_active_learning/{test_name}_var_samples.txt')
    np.savetxt(txt_file, var_samples, delimiter=',')
    
    # estimate posterior using grip estimation based on final GP emulator
    grid_estimator = GridEstimation(
        ndim=al_variance.ndim,
        bounds=al_variance.bounds,
        ln_pxl=al_variance.approx_ln_pxl)
    
    posterior, x_ndim = grid_estimator.run(nbins)   

    # estimate posterior using mh estimation based on final GP emulator
    mh_estimator = MetropolisHastingsEstimation(
        ndim=al_variance.ndim,
        bounds=al_variance.bounds,
        ln_pxl=al_variance.approx_ln_pxl)

    init_state = np.array([-4,-4])
    f_sample = multivariate_normal.rvs
    nburn = 100
    nthin = 10
    kwgs_f_sample = {'random_state': np.random.default_rng(test_entropy_mh_seed)}

    mh_sampler = MetropolisHastings(
        ndim=al_variance.ndim,
        init_state=init_state,
        f_sample=f_sample,
        bounds=al_variance.bounds,
        nburn=nburn,
        nthin=nthin,
        seed=test_entropy_mh_seed,
        kwgs_f_sample=kwgs_f_sample)
    
    nsamples = 5000
    mh_samples, mh_accept = mh_estimator.run(nsamples, mh_sampler)

    assert mh_samples.shape == (nsamples, al_variance.ndim)
    assert mh_accept.shape == (nsamples,)
    print(f'{test_name} mh acceptance ratio: ', np.mean(mh_accept))

    # plot
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(init_var_samples[:,0], init_var_samples[:,1], s=10, c='r',
        marker='o', zorder=1, alpha=0.8)
    axes[0].scatter(var_samples[n0:,0], var_samples[n0:,1], s=15, c='k',
        marker='+', zorder=2, alpha=0.8)
    
    posterior = np.where(posterior < 1e-10, np.nan, posterior)

    contour = axes[0].contour(x_ndim[0], x_ndim[1], np.transpose(posterior),
        levels=10, zorder=0)
    plt.colorbar(contour, ax=axes[0])
    
    axes[0].set_title(
        'Active learning \n prior U[-5,5]x[-5,5] \n simulator y=x \n'
        'likelihood Rosenbrock \n '+
        f'n0={n0}, actual_niter={len(var_samples)-n0}' + '\n' +
        f'{test_name} grid approximation')
    
    axes[1].scatter(mh_samples[:, 0],mh_samples[:,1], s=10, c='r',
        marker='o', alpha=0.1)
    axes[1].set_title(
        'Active learning \n prior U[-5,5]x[-5,5] \n simulator y=x \n'
        'likelihood Rosenbrock \n '+
        f'n0={n0}, actual_niter={len(var_samples)-n0}' + '\n' +
        f'{test_name} metropolis hastings')
    
    for i in range(2):
        axes[i].set_xlim(-5.1, 5.1)
        axes[i].set_ylim(-5.1, 5.1)
        axes[i].set_xlabel('x1')
        axes[i].set_ylabel('x2')

    png_file = os.path.join(dir_test,
        f'temp_active_learning/{test_name}.png')
    plt.savefig(png_file, bbox_inches='tight')

    assert os.path.exists(png_file)