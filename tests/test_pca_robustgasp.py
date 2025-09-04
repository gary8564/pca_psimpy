import copy
import pytest
import matplotlib
# Use a non-interactive backend to avoid rendering plots during tests
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psimpy.emulator.pca import LinearPCA, NonlinearPCA, InputDimReducer, OutputDimReducer
from psimpy.emulator.pca_robustgasp import PCAScalarGaSP, PCAPPGaSP

# Suppress plot.show() calls in PCA methods
@pytest.fixture(autouse=True)
def no_plot(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def reduced_dim(reducer: InputDimReducer | None, X) -> int:
    """Helper function to obtain the input dimension

    Args:
        reducer (InputDimReducer | None): Dimensionality reduction
        X (np.ndarray): Input dataset

    Returns:
        int: number of input dimensions
    """
    if reducer is None:
        return X.shape[1]
    
    r = copy.deepcopy(reducer)
    r.reducer.fit(X, show_cum_var_plot=False)
    n_comp = r.reducer.n_components
    return int(n_comp)

# PCAScalarGaSP – scalar output, with/without input PCA   
@pytest.mark.parametrize(
    "use_input_pca, n_components",
    [(False, 0.90), (True, 3)],
)  
def test_pca_scalar_gasp(humanity_data, use_input_pca, n_components):
    X_train, Y_train, X_test, _ = humanity_data
    y_train = Y_train[:, 0]
    pca = LinearPCA(n_components=n_components)
    input_reducer = (
        InputDimReducer(pca) 
        if use_input_pca 
        else None
    )
    n_dim = reduced_dim(input_reducer, X_train)
    
    model = PCAScalarGaSP(
        ndim=n_dim,
        input_dim_reducer=input_reducer,
    )
    _ = model.train(X_train, y_train)
    preds, _ = model.predict(X_test)
    assert preds.shape == (X_test.shape[0], 4)   
    samples = model.sample(X_test, nsamples=8)
    assert samples.shape == (X_test.shape[0], 8)

# PCAPPGaSP - direct prediction without MC sampling
@pytest.mark.parametrize(
    "input_pca_cfg, output_pca_cfg",
    [
        # (Input reducer, Output reducer)
        (LinearPCA(n_components=3), LinearPCA(n_components=2)),                   # linear / linear
        (LinearPCA(n_components=3), NonlinearPCA(n_components=2, kernel="rbf",
                                                 fit_inverse_transform=True)),    # linear / kernel
        (None,                      LinearPCA(n_components=2)),                   # no input PCA
    ],
)
def test_pca_ppgasp_predict(humanity_data, input_pca_cfg, output_pca_cfg):
    X_train, Y_train, X_test, _ = humanity_data
    input_reducer  = InputDimReducer(input_pca_cfg) if input_pca_cfg else None
    output_reducer = OutputDimReducer(output_pca_cfg)
    model = PCAPPGaSP(
        ndim=reduced_dim(input_reducer, X_train),
        input_dim_reducer=input_reducer,
        output_dim_reducer=output_reducer,
    )
    _ = model.train(X_train, Y_train)
    _, preds, _ = model.predict(X_test)           
    assert preds.shape == (X_test.shape[0], Y_train.shape[1])

def test_ppgasp_predict_without_inverse(humanity_data):
    """
    NonlinearPCA with fit_inverse_transform=False should raise error.
    """
    X_train, Y_train, X_test, _ = humanity_data
    pca = LinearPCA(n_components=3)
    input_reducer = InputDimReducer(pca)
    output_reducer = OutputDimReducer(
        NonlinearPCA(n_components=2, kernel="rbf", fit_inverse_transform=False)
    )

    model = PCAPPGaSP(
        ndim=reduced_dim(input_reducer, X_train),
        input_dim_reducer=input_reducer,
        output_dim_reducer=output_reducer,
    )
    _ = model.train(X_train, Y_train)

    with pytest.raises(NotImplementedError):
        _ = model.predict(X_test)