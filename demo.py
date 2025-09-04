import numpy as np
import copy
import os
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import plotting_extent
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from psimpy.emulator.pca import LinearPCA, NonlinearPCA, InputDimReducer, OutputDimReducer
from psimpy.emulator.pca_robustgasp import PCAScalarGaSP, PCAPPGaSP

def zero_truncated_data(filepath, threshold, valid_cols=None):
        """ Preprocess the dataset to filter out the zeros so that GP emulators can be trained
        
        Args:
            filepath (str): File path of the dataset to be preprocessed.
            threshold (int, float): Threshold value to define valid cells from simulations.
            valid_cols (list, optional): column numbers to extract. Defaults to None.
        
        Raises:
            TypeError: threshold must be a number
            ValueError: threshold cannot be negative
        
        Returns:
            training (np.ndarray): A data frame consisting of the vector outputs from simulations
            valid_cols (np.ndarray): An array consisting of the valid column names
        """
        if not isinstance(threshold, (int, float)):
            raise TypeError('threshold must be a number')
        if threshold < 0:
            raise ValueError('threshold cannot be negative')
    
        with rasterio.open(filepath) as src:
            rows = src.height
            cols = src.width
            size = src.count
        
            unstacked = np.zeros((size, rows * cols))
        
            for sim in range(size):
                unstacked[sim, :] = src.read(sim + 1).reshape(1, rows * cols)
        
        if valid_cols is None:
            valid_cols = np.where(unstacked >= threshold, 1, 0).sum(axis=0)
        indices = np.flatnonzero(valid_cols)
        nz_out = unstacked[:, indices]
        return nz_out, valid_cols, rows, cols

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

def reconstruct_output_image(output, rows, cols, valid_cols):
    """Reconstruct the output data to original 2D image array

    Args:
        output (np.ndarray): flattened image array 
        rows (int): number of rows of the original 2D image array
        cols (int): number of columns of the original 2D image array
        valid_cols (np.ndarray): the valid column indices stored in zero-truncated preprocessing step

    Returns:
        output_hmax: reconstructed original image array (shape: (num_samples, rows, cols))
        output_mean: mean value of image array over the number of data samples (shape: (rows, cols))
        output_std: standard deviation of image array over the number of data samples( shape: (rows, cols))
    """
    output_size = output.shape[0]
    indices = np.flatnonzero(valid_cols)
    output_index = [int(i) for i in list(indices)]
    reconstructed_output = np.zeros((output_size, rows * cols))
    reconstructed_output[:,output_index] = output
    output_mean = reconstructed_output.mean(axis=0).reshape(rows, cols)
    output_std = reconstructed_output.std(axis=0).reshape(rows, cols)
    output_hmax = reconstructed_output.reshape((-1, rows, cols))
    print("Reconstructed output image dimension: ", output_hmax.shape)
    return output_hmax, output_mean, output_std

def viz_output_image(hill_path, mean_gt, std_gt, mean_pred, std_pred) -> None:
    """Visualize the reconstructed output image

    Args:
        hill_path (str): filepath of the background image
        mean_gt (np.ndarray): mean value of the ground-truth image array
        std_gt (np.ndarray): standard deviation value of the ground-truth image array
        mean_pred (np.ndarray): mean value of the predicted image array
        std_pred (np.ndarray): standard deviation value of the predicted image array
    """
    mean_gt_mask = np.ma.masked_where(mean_gt < 0.1, mean_gt, copy=True)
    std_gt_mask = np.ma.masked_where(mean_gt < 0.1, std_gt, copy=True)
    mean_pred_mask = np.ma.masked_where(mean_pred < 0.1, mean_pred, copy=True)
    std_pred_mask = np.ma.masked_where(mean_pred < 0.1, std_pred, copy=True)
    with rasterio.open(hill_path, 'r') as hill:
        hill_arr = hill.read(1)
    hill_mask = np.ma.masked_where(hill_arr < -30000, hill_arr, copy=True)
    diff_mean = mean_pred - mean_gt
    diff_std = std_pred - std_gt
    diff_mean_mask = np.ma.masked_where(diff_mean == 0, diff_mean, copy=True)
    diff_std_mask = np.ma.masked_where(diff_std == 0, diff_std, copy=True)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
        nrows=2, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1]}
    )
    ax1.imshow(hill_mask, cmap='Greys', extent=plotting_extent(hill))
    c1 = ax1.imshow(
        mean_gt_mask, cmap='viridis', extent=plotting_extent(hill), zorder=1, vmin=0.1, vmax=80
    )
    fig.colorbar(
        c1, ax=ax1, location='top', orientation='horizontal', label='Mean flow height [m]'
    )
    ax1.set_ylabel('Northing [x 10$^6$ m]')
    ax1.set_xticks(ticks=np.arange(1489000, 1493600, 1000), labels=None)
    ax1.set_yticks(
        ticks=np.arange(5201000, 5205001, 1000), labels=[5.201, 5.202, 5.203, 5.204, 5.205]
    )
    ax1.text(1488500, 5207000, 'A', weight='bold')
    ax1.axes.get_xaxis().set_ticklabels([])
    
    ax2.imshow(hill_mask, cmap='Greys', extent=plotting_extent(hill))
    c2 = ax2.imshow(mean_pred_mask, cmap='viridis', extent=plotting_extent(hill), vmin=0.1, vmax=80)
    fig.colorbar(
        c2, ax=ax2, location='top', orientation='horizontal', label='Mean flow height [m]'
    )
    ax2.set_xticks(
        ticks=np.arange(1489000, 1493600, 1000), labels=[1.498, '1.490', 1.491, 1.492, 1.493]
    )
    ax2.set_yticks(
        ticks=np.arange(5201000, 5205001, 1000), labels=[5.201, 5.202, 5.203, 5.204, 5.205]
    )
    ax2.text(1488500, 5207000, 'B', weight='bold')
    ax2.axes.get_yaxis().set_ticklabels([])
    ax2.axes.get_xaxis().set_ticklabels([])
    ax3.imshow(hill_mask, cmap='Greys', extent=plotting_extent(hill))
    c3 = ax3.imshow(diff_mean_mask, cmap='RdBu', extent=plotting_extent(hill), vmin=-4, vmax=4)
    fig.colorbar(
        c3,
        ax=ax3,
        location='top',
        orientation='horizontal',
        label='Difference in mean flow height [m]',
    )
    ax3.set_xticks(
        ticks=np.arange(1489000, 1493600, 1000), labels=[1.498, '1.490', 1.491, 1.492, 1.493]
    )
    ax3.set_yticks(
        ticks=np.arange(5201000, 5205001, 1000), labels=[5.201, 5.202, 5.203, 5.204, 5.205]
    )
    ax3.text(1488500, 5207000, 'C', weight='bold')
    ax3.axes.get_yaxis().set_ticklabels([])
    ax3.axes.get_xaxis().set_ticklabels([])
    
    ax4.imshow(hill_mask, cmap='Greys', extent=plotting_extent(hill))
    c4 = ax4.imshow(
        std_gt_mask, cmap='viridis', extent=plotting_extent(hill), zorder=1, vmin=0.1, vmax=40
    )
    fig.colorbar(
        c4,
        ax=ax4,
        location='top',
        orientation='horizontal',
        label='Std. deviation flow height [m]',
    )
    ax4.set_xticks(
        ticks=np.arange(1489000, 1493600, 1000), labels=[1.498, '1.490', 1.491, 1.492, 1.493]
    )
    ax4.set_yticks(
        ticks=np.arange(5201000, 5205001, 1000), labels=[5.201, 5.202, 5.203, 5.204, 5.205]
    )
    ax4.text(1488500, 5207000, 'D', weight='bold')
    ax4.set_xlabel('Easting [x 10$^6$ m]')
    ax4.set_ylabel('Northing [x 10$^6$ m]')
    ax5.imshow(hill_mask, cmap='Greys', extent=plotting_extent(hill))
    c5 = ax5.imshow(std_pred_mask, cmap='viridis', extent=plotting_extent(hill), vmin=0.1, vmax=40)
    fig.colorbar(
        c5,
        ax=ax5,
        location='top',
        orientation='horizontal',
        label='Std. deviation flow height [m]',
    )
    ax5.set_xticks(
        ticks=np.arange(1489000, 1493600, 1000), labels=[1.498, '1.490', 1.491, 1.492, 1.493]
    )
    ax5.set_yticks(
        ticks=np.arange(5201000, 5205001, 1000), labels=[5.201, 5.202, 5.203, 5.204, 5.205]
    )
    ax5.text(1488500, 5207000, 'E', weight='bold')
    ax5.axes.get_yaxis().set_ticklabels([])
    ax5.set_xlabel('Easting [x 10$^6$ m]')
    
    ax6.imshow(hill_mask, cmap='Greys', extent=plotting_extent(hill))
    c6 = ax6.imshow(diff_std_mask, cmap='RdBu', extent=plotting_extent(hill), vmin=-20, vmax=20)
    fig.colorbar(
        c6,
        ax=ax6,
        location='top',
        orientation='horizontal',
        label='Difference in std. deviation [m]',
    )
    ax6.set_xticks(
        ticks=np.arange(1489000, 1493600, 1000), labels=[1.498, '1.490', 1.491, 1.492, 1.493]
    )
    ax6.set_yticks(
        ticks=np.arange(5201000, 5205001, 1000), labels=[5.201, 5.202, 5.203, 5.204, 5.205]
    )
    ax6.text(1488500, 5207000, 'F', weight='bold')
    ax6.axes.get_yaxis().set_ticklabels([])
    ax6.set_xlabel('Easting [x 10$^6$ m]')
    plt.rcParams['figure.figsize'] = [18 / 2.54, 18 / 2.54]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()

# Load data
curr_path = os.path.dirname(os.path.abspath(__file__))
data_root_folder = os.path.join(curr_path, "tests", "data", "acheron")
qoi = "hmax"
hill_path = os.path.join(data_root_folder, "hillshade_acheron.tif")
train_input_filepath = os.path.join(data_root_folder, "input.csv")
test_input_filepath = os.path.join(data_root_folder, "input_test.csv")
train_X = np.genfromtxt(train_input_filepath, delimiter=',', skip_header=1)
test_X = np.genfromtxt(test_input_filepath, delimiter=',', skip_header=1)
train_output_filepath = os.path.join(data_root_folder, "output", qoi + "_stack.tif") 
test_output_filepath = os.path.join(data_root_folder, "output_test", qoi + "_stack.tif") 
train_Y, valid_cols, train_rows, train_cols = zero_truncated_data(train_output_filepath, threshold=0.5)
test_Y, _, test_rows, test_cols = zero_truncated_data(test_output_filepath, threshold=0.5, valid_cols=valid_cols) 
assert (train_rows == test_rows) and (train_cols == test_cols), "The row and column of the output in training and test dataset must be consistent."
print(f"Train_X: {train_X.shape}, Train_Y: {train_Y.shape}")
print(f"Test_X: {test_X.shape}, Test_Y: {test_Y.shape}")

# Define the dimensionality reduction and Gaussian Process (GP) model
input_reducer  = None
# output_pca_model = LinearPCA()
# output_pca_model._compute_n_components(train_Y)
output_pca_model = NonlinearPCA(n_components=7, alpha=0.0001)
output_reducer = OutputDimReducer(output_pca_model)
model = PCAPPGaSP(
    ndim=reduced_dim(input_reducer, train_X),
    input_dim_reducer=input_reducer,
    output_dim_reducer=output_reducer,
)

# Train the GP model
_ = model.train(train_X, train_Y)

# Inference
predictions_latent, predictions_original, _ = model.predict(test_X)         
rmse = root_mean_squared_error(predictions_original.flatten(), test_Y.flatten())    
print("RMSE: ", rmse)

# Visualization
gt_hmax, gt_mean, gt_std = reconstruct_output_image(test_Y, test_rows, test_cols, valid_cols)
pred_hmax, pred_mean, pred_std = reconstruct_output_image(predictions_original, test_rows, test_cols, valid_cols)
viz_output_image(hill_path, gt_mean, gt_std, pred_mean, pred_std)