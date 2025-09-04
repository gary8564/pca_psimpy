import numpy as np
import pytest
import matplotlib
# Use a non-interactive backend to avoid rendering plots during tests
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psimpy.emulator.pca import BaseReducer, LinearPCA, NonlinearPCA, InputDimReducer, OutputDimReducer

# Suppress plot.show() calls in PCA methods
@pytest.fixture(autouse=True)
def no_plot(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

class DummyReducer(BaseReducer):
    """A simple reducer used for testing BaseReducer interface."""
    def __init__(self):
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'fitted') or not self.fitted:
            raise RuntimeError("Reducer not fitted.")
        return X + 1

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        return X_reduced - 1

# Tests for BaseReducer using DummyReducer
def test_dummy_reducer_interface():
    X = np.array([[1, 2], [3, 4]])
    dr = DummyReducer()
    # transform before fit should raise
    with pytest.raises(RuntimeError):
        dr.transform(X)
    # fit_transform should fit and transform
    out = dr.fit_transform(X)
    assert np.array_equal(out, X + 1)
    # transform after fit
    out2 = dr.transform(X)
    assert np.array_equal(out2, X + 1)
    # inverse_transform
    inv = dr.inverse_transform(out)
    assert np.array_equal(inv, X)

# Tests for Linear PCA 

@pytest.mark.parametrize("invalid_n", [
    [1, 2, 3],   # wrong type
    0,           # int <= 0
    "invalid", # unsupported string
    -0.5,        # float <= 0
    1.5          # float >= 1
])
def test_linear_pca_invalid_n_components(invalid_n):
    """
    Expect initialization to raise for invalid n_components values.
    """
    with pytest.raises((TypeError, AttributeError)):
        LinearPCA(n_components=invalid_n)

@pytest.mark.parametrize("n_comp, should_raise", [
    (3, False),       # explicit component count
    (0.975, False),   # explained variance threshold
    ("mle", False), # MLE selection
    (None, True)      # None is not supported
])
def test_linear_pca_init_behavior(n_comp, should_raise):
    """
    Test that constructor either creates a model or raises appropriately.
    """
    if should_raise:
        with pytest.raises(TypeError):
            LinearPCA(n_components=n_comp)
    else:
        pca = LinearPCA(n_components=n_comp)
        assert isinstance(pca, LinearPCA)

@pytest.mark.parametrize("n_comp", [3, 0.90, "mle"] )
def test_linear_pca_fit_transform_and_inverse(humanity_data, n_comp):
    X, _, _, _ = humanity_data

    pca = LinearPCA(n_components=n_comp)

    with pytest.raises(RuntimeError):
        pca.transform(X)

    X_reduced = pca.fit_transform(X)

    # Reduced dimensions should match n_components logic
    expected_dim = pca.n_components
    assert 1 <= pca.n_components <= min(X.shape), f"Number of components must be lower than min(X.shape), but got {pca.n_components}"
    assert 0 < pca.explained_variance < 1, f"Explained variance must be in between 0 and 1, but got {pca.explained_variance}"
    if isinstance(n_comp, int):
        assert expected_dim == n_comp, f"The expected dimension {expected_dim} is mismatched with specified n_comp {n_comp}."
    elif isinstance(n_comp, float):
        assert expected_dim == pca._compute_n_components(X, n_comp, show_cum_var_plot=False), "The expected dimension is mismatched with pca._compute_n_components() returned value."
    else:
        assert isinstance(pca.explained_variance, float)
    assert X_reduced.shape == (X.shape[0], expected_dim)

    # Test inverse_transform works and returns correct shape
    X_original = pca.inverse_transform(X_reduced)
    assert X_original.shape == X.shape
    mse = np.mean((X - X_original) ** 2)
    # allow error proportional to discarded variance
    if isinstance(n_comp, float):
        retained = n_comp
    else:
        retained = expected_dim / X.shape[1]
    orig_var = np.var(X, axis=0).sum()
    max_mse = (1.0 - retained) * orig_var + 1e-8
    assert mse < max_mse, f"Reconstruction error {mse} exceeds maximum allowed {max_mse} for retained variance {retained}"

# Nonlinear PCA
@pytest.mark.parametrize("kernel", ["rbf", "sigmoid", "poly"])
def test_nonlinear_pca_fit_transform_and_inverse(humanity_data, kernel):
    X, _, _, _ = humanity_data
    kpca = NonlinearPCA(n_components=4, kernel=kernel, fit_inverse_transform=True)
    X_reduced = kpca.fit_transform(X)
    assert X_reduced.shape == (X.shape[0], 4)
    X_rec = kpca.inverse_transform(X_reduced)
    assert X_rec.shape == X.shape, "Reconstruction should have original feature dimension."

def test_nonlinear_pca_no_inverse():
    X = np.random.randn(20, 5)
    kpca = NonlinearPCA(n_components=3, kernel="rbf", fit_inverse_transform=False)
    kpca.fit(X)
    X_red = kpca.transform(X)
    with pytest.raises(NotImplementedError):
        kpca.inverse_transform(X_red)

def test_nonlinear_pca_invalid_kernel():
    with pytest.raises(ValueError):
        NonlinearPCA(n_components=2, kernel="unsupported")

# InputDimReducer / OutputDimReducer
def test_input_dim_reducer_behavior():
    X = np.array([[1, 2], [3, 4]])
    reducer = InputDimReducer(DummyReducer())
    with pytest.raises(ValueError):
        reducer.fit(np.array([]))
    reducer.fit(X)
    # transform before fit triggers underlying error
    with pytest.raises(RuntimeError):
        InputDimReducer(DummyReducer()).transform(X)
    Y = reducer.transform(X)
    assert np.array_equal(Y, X + 1)
    Z = reducer.fit_transform(X)
    assert np.array_equal(Z, X + 1)

def test_output_dim_reducer_behavior():
    Y = np.array([[5, 6], [7, 8]])
    reducer = OutputDimReducer(DummyReducer())
    with pytest.raises(ValueError):
        reducer.fit(np.array([]))
    reducer.fit(Y)
    Xr = reducer.transform(Y)
    assert np.array_equal(Xr, Y + 1)
    Z = reducer.fit_transform(Y)
    assert np.array_equal(Z, Y + 1)
    X_rec = reducer.inverse_transform(Z)
    assert np.array_equal(X_rec, Y)

def test_input_and_output_reducers(humanity_data):
    X, Y, *_ = humanity_data
    input_reduced = InputDimReducer(LinearPCA(n_components=2))
    output_reduced = OutputDimReducer(LinearPCA(n_components=3))

    X_r = input_reduced.fit_transform(X)
    Y_r = output_reduced.fit_transform(Y)

    # revert back to original output dimension space
    Y_back = output_reduced.inverse_transform(Y_r)
    assert Y_back.shape == Y.shape

    # Input reducer cannot "inverse‑transform" (no method exposed)
    with pytest.raises(AttributeError):
        _ = input_reduced.inverse_transform  

