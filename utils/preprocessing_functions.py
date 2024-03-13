# from umap import UMAP  # Import UMAP from umap module
from scipy.interpolate import interp1d
import pandas as pd
import os
import h5py
import numpy as np
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # Import tqdm for progress bar
from sklearn.manifold import TSNE
#from skimage.feature import peak_local_max
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
import matplotlib as mpl
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


def preprocess_data(directory_path, known_distance_mm, stim_types, group_types):
    directory_files = os.listdir(directory_path)
    locations = []
    session_ends = []
    stims = []
    groups = []

    def fill_missing(arr):
        # Function to fill missing values in an array
        arr[arr == 0] = np.nan
        return arr

    for loaded_file in directory_files:
        filename = os.path.join(directory_path, loaded_file)

        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            file_locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]

            # Step 1: Calculate the pixel distance between the reference objects
            ref_object_3 = file_locations[:, 3, :, 0]
            ref_object_4 = file_locations[:, 4, :, 0]
            distance_pixels = np.linalg.norm(ref_object_3 - ref_object_4, axis=1)

            # Step 2: Calculate the conversion factor from pixels to millimeters
            pixels_to_mm = known_distance_mm / distance_pixels

            # Step 3: Scale the positions of objects [:,0,:,0], [:,1,:,0], and [:,2,:,0]
            # based on the calculated conversion factor
            objects_to_scale = file_locations[:, :3, :, 0]  # Selecting [:,0,:,0], [:,1,:,0], and [:,2,:,0]

            # Scale each coordinate separately
            for i in range(objects_to_scale.shape[1]):  # Iterate over the nodes
                for j in range(objects_to_scale.shape[2]):  # Iterate over the axes (x and y)
                    objects_to_scale[:, i, j] *= pixels_to_mm

            file_locations[:, :3, :, 0] = objects_to_scale

            # Invert y axis
            file_locations[:, 0, 1, :] = -file_locations[:, 0, 1, :]
            file_locations[:, 1, 1, :] = -file_locations[:, 1, 1, :]
            file_locations[:, 2, 1, :] = -file_locations[:, 2, 1, :]

            first_x_value = file_locations[0, :, 0, :]
            slice_0 = fill_missing(file_locations[:, 0, 1, 0])
            slice_1 = fill_missing(file_locations[:, 1, 1, 0])
            slice_2 = fill_missing(file_locations[:, 2, 1, 0])

            # Find the minimum value among the slices
            min_value = min(np.nanmin(slice_0), np.nanmin(slice_1), np.nanmin(slice_2))

            # Find the frame number where the minimum value occurs
            if min_value in slice_0:
                frame_number = np.where(slice_0 == min_value)[0][0]
                lowest_bp = 0
            elif min_value in slice_1:
                frame_number = np.where(slice_1 == min_value)[0][0]
                lowest_bp = 1
            else:
                frame_number = np.where(slice_2 == min_value)[0][0]
                lowest_bp = 2

            translated_locations = np.zeros_like(file_locations)

            translated_locations[:, :, 0, :] = file_locations[:, :, 0, :] - first_x_value
            translated_locations[:, :, 1, :] = file_locations[:, :, 1, :] - file_locations[frame_number, lowest_bp, 1, 0]

            if not session_ends:  # Check if the list is empty
                session_ends.append(translated_locations.shape[0])
            else:
                session_ends.append(session_ends[-1] + translated_locations.shape[0])

            # Check if any stim_type is present in the loaded_file
            stim_found = False
            for stim in stim_types:
                if stim in loaded_file:
                    stims.append(stim)
                    stim_found = True
                    break
            if not stim_found:
                stims.append("NA")

            # Check if any group_type is present in the loaded_file
            group_found = False
            for group in group_types:
                if group in loaded_file:
                    groups.append(group)
                    group_found = True
                    break
            if not group_found:
                groups.append("NA")

        locations.append(translated_locations)

    locations = np.concatenate(locations, axis=0)
    return locations, session_ends, stims, groups


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y



def rolling_median(Y, window_size=10):
    """Computes rolling median independently along each dimension after the first."""

    # Store initial shape
    initial_shape = Y.shape

    # Flatten after first dim
    Y = Y.reshape((initial_shape[0], -1))

    X = np.zeros_like(Y)

    # Apply rolling median along each slice
    for i in range(Y.shape[-1]):
        x = Y[:, i]
        series = pd.Series(x)

        # Compute rolling median with a specified window size
        # 'min_periods=1' ensures that the median is computed even with fewer elements
        rolling_median = series.rolling(window=window_size, min_periods=1, center=True).median()
        # rolling_median = series.rolling(window_size).mean()

        # Save the rolling median values
        X[:, i] = rolling_median

    # Restore to initial shape
    X = X.reshape(initial_shape)

    return X


def rolling_mean(Y, window_size=10):
    """Computes rolling median independently along each dimension after the first."""

    # Store initial shape
    initial_shape = Y.shape

    # Flatten after first dim
    Y = Y.reshape((initial_shape[0], -1))

    X = np.zeros_like(Y)

    # Apply rolling median along each slice
    for i in range(Y.shape[-1]):
        x = Y[:, i]
        series = pd.Series(x)

        # Compute rolling median with a specified window size
        # 'min_periods=1' ensures that the median is computed even with fewer elements
        # rolling_median = series.rolling(window=window_size, min_periods=1, center=True).median()
        rolling_median = series.rolling(window_size).mean()

        # Save the rolling median values
        X[:, i] = rolling_median

    # Restore to initial shape
    X = X.reshape(initial_shape)

    return X



def smooth_diff(series, window_size):
    # Compute differences between consecutive points
    diffs = np.diff(series)

    # Assuming time interval is 1 unit
    time_diff = 1

    # Compute velocities
    velocities = diffs / time_diff

    # Smooth velocities using rolling window
    smoothed_velocities = np.convolve(velocities, np.ones(window_size)/window_size, mode='same')

    # Pad with NaN to match the length of the original series
    padding = window_size // 2
    smoothed_velocities = np.concatenate((np.full(padding, np.nan), smoothed_velocities, np.full(padding, np.nan)))

    # Trim to match the length of the original series
    smoothed_velocities = smoothed_velocities[:len(series)]

    smoothed_velocities = np.nan_to_num(smoothed_velocities, nan=0)

    return smoothed_velocities


def compound_smoother(locations, window_size):
    locations = fill_missing(locations)
    rolling_med = rolling_median(locations, window_size=window_size)
    roll_mean = rolling_mean(rolling_med, window_size=window_size)
    roll_mean = fill_missing(roll_mean)
    
    return roll_mean


def calculate_direction_rolling(x_positions, y_positions, window):
    """
    Calculate the direction of travel with respect to the x-axis using rolling positions,
    padding the array to keep the shape the same.

    Args:
    - x_positions: Array or list containing the x positions.
    - y_positions: Array or list containing the y positions.
    - window: The window size for rolling positions.

    Returns:
    - direction: Array of angles (in radians) representing the direction of travel.
    """
    direction = []

    # Extend the range to include the padding
    for i in range(len(x_positions) - window + 1):
        # Calculate the direction (angle) between consecutive positions
        dx = x_positions[i+window-1] - x_positions[i]
        dy = y_positions[i+window-1] - y_positions[i]
        angle = np.arctan2(dy, dx)
        direction.append(angle)

    # Pad the array at the end with NaN values
    direction += [np.nan] * (len(x_positions) - len(direction))

    return np.array(direction)


def instance_node_velocities(my_locations, session_ends, window_size = 3):
    frame_count = my_locations.shape[0]  # Assuming frame_count is defined elsewhere
    node_locations = my_locations[:, :, :, 0]
    node_velocities = np.zeros((frame_count, 20))  # Increase the array size to accommodate the new entry

    # take into account relative velocities of points
    node_velocities[:, 0] = smooth_diff(node_locations[:, 0, 0], window_size = window_size)
    node_velocities[:, 1] = smooth_diff(node_locations[:, 1, 0], window_size = window_size)
    node_velocities[:, 2] = smooth_diff(node_locations[:, 2, 0], window_size = window_size)

    node_velocities[:, 3] = smooth_diff(node_locations[:, 0, 1], window_size = window_size)
    node_velocities[:, 4] = smooth_diff(node_locations[:, 1, 1], window_size = window_size)
    node_velocities[:, 5] = smooth_diff(node_locations[:, 2, 1], window_size = window_size)

    # acceleration (I think)
    node_velocities[:, 6] = smooth_diff(node_velocities[:, 0], window_size = window_size)
    node_velocities[:, 7] = smooth_diff(node_velocities[:, 1], window_size = window_size)
    node_velocities[:, 8] = smooth_diff(node_velocities[:, 2], window_size = window_size)
    node_velocities[:, 9] = smooth_diff(node_velocities[:, 3], window_size = window_size)
    node_velocities[:, 10] = smooth_diff(node_velocities[:, 4], window_size = window_size)
    node_velocities[:, 11] = smooth_diff(node_velocities[:, 5], window_size = window_size)

    # take into account height above ground
    node_velocities[:, 12] = node_locations[:, 0, 1]
    node_velocities[:, 13] = node_locations[:, 1, 1]
    node_velocities[:, 14] = node_locations[:, 2, 1]

    # take into account lateral distance from starting point
    node_velocities[:, 15] = abs(node_locations[:, 0, 0] - node_locations[1, 0, 0])
    node_velocities[:, 16] = abs(node_locations[:, 1, 0] - node_locations[1, 1, 0])
    node_velocities[:, 17] = abs(node_locations[:, 2, 0] - node_locations[1, 2, 0])

    # height difference between toe and heel
    node_velocities[:, 18] = node_locations[:, 0, 1] - node_locations[:, 2, 1]


    node_velocities[:, 19] = np.linalg.norm(node_locations[:, 0, :] - node_locations[:, 2, :], axis=1)

    ############# EXPERIMENTAL OUTLIER DELETION

    for i in range(min(18, node_velocities.shape[1])):
        for session_end in session_ends:
            start_index = max(0, session_end - window_size)
            end_index = min(node_velocities.shape[0], session_end + window_size + 1)
            # Get the value from just before the window
            value_before_window = node_velocities[max(0, start_index - 1), i]
            # Assign this value to the entire window for the feature i
            node_velocities[start_index:end_index, i] = value_before_window

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(node_velocities)

    # Transform training data
    node_velocities = scaler.transform(node_velocities)

    # Store normalization parameters
    min_train = scaler.data_min_
    max_train = scaler.data_max_

    return node_velocities, min_train, max_train

def create_sliding_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        windows.append(window)
    return np.array(windows)

