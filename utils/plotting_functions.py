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
from scipy.interpolate import griddata
import datetime



def plot_withdrawal_features(node_velocities, node_features, vmin=-1, vmax=1):
    plt.figure(figsize=(20, 8))
    plt.imshow(node_velocities.T, aspect='auto', interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.xlabel('Frames')
    plt.yticks(np.arange(20), node_features)
    plt.title('Normalized Paw Withdrawal Features')
    plt.show()


def plot_paw_trajectory(locations, smoothed_locations, toe_index, center_index, heel_index):
    TOE_INDEX = toe_index
    CENTER_INDEX = center_index
    HEEL_INDEX = heel_index

    toe_loc = locations[:, TOE_INDEX, :, :]
    center_loc = locations[:, CENTER_INDEX, :, :]
    heel_loc = locations[:, HEEL_INDEX, :, :]

    toe_mean = smoothed_locations[:, TOE_INDEX, :, :]
    center_mean = smoothed_locations[:, CENTER_INDEX, :, :]
    heel_mean = smoothed_locations[:, HEEL_INDEX, :, :]

    sns.set('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15,6]

    plt.figure()
    plt.plot(toe_loc[:,1,0], 'y',label='toe_raw')
    plt.plot(toe_mean[:,1,0], 'y',label='toe_smooth', linestyle = 'dashed')
    plt.plot(center_loc[:,1,0], 'r',label='center_raw')
    plt.plot(center_mean[:,1,0], 'r',label='center_smooth', linestyle = 'dashed')
    plt.plot(heel_loc[:,1,0], 'g',label='heel_raw')
    plt.plot(heel_mean[:,1,0], 'g',label='heel_smooth', linestyle = 'dashed')
    plt.legend(loc="center right")
    plt.title('Paw Y-axis raw trajectory')

    plt.figure()
    plt.plot(toe_loc[:,0,0], 'y',label='toe_raw')
    plt.plot(toe_mean[:,0,0], 'y',label='toe_smooth', linestyle = 'dashed')
    plt.plot(center_loc[:,0,0], 'r',label='center_raw')
    plt.plot(center_mean[:,0,0], 'r',label='center_smooth', linestyle = 'dashed')
    plt.plot(heel_loc[:,0,0], 'g',label='heel_raw')
    plt.plot(heel_mean[:,0,0], 'g',label='heel_smooth', linestyle = 'dashed')
    plt.legend(loc="center right")
    plt.title('Paw X-axis raw trajectory')


def plot_displacement_vectors(embedding_data, static_threshold = 40, grid_rows = 15, grid_cols = 15):
    # Example x-y data (replace this with your own data)
    x_data = embedding_data[:, 0]
    y_data = embedding_data[:, 1]
    threshold_percentage = static_threshold  # Set the threshold percentage here (for example, 40%)

    # Check for NaN values in the input data
    if np.isnan(np.sum(x_data)) or np.isnan(np.sum(y_data)):
        print("Error: Input data contains NaN values.")
    else:
        # Define the grid dimensions
        num_rows = grid_rows
        num_cols = grid_cols

        # Calculate the range of the x and y data
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

        # Extend the x and y limits by 10%
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_extend = 0.1 * x_range
        y_extend = 0.1 * y_range
        x_min -= x_extend
        x_max += x_extend
        y_min -= y_extend
        y_max += y_extend

        # Calculate the grid cell size based on the extended bounds
        grid_width = (x_max - x_min) / num_cols
        grid_height = (y_max - y_min) / num_rows

        # Initialize arrays to store average displacement vectors
        avg_displacement_x = np.zeros((num_rows, num_cols))
        avg_displacement_y = np.zeros((num_rows, num_cols))
        counts = np.zeros((num_rows, num_cols), dtype=int)

        # Compute average displacement vectors for each grid cell
        for i in range(len(x_data)):
            row_index = min(int((y_data[i] - y_min) // grid_height), num_rows - 1)
            col_index = min(int((x_data[i] - x_min) // grid_width), num_cols - 1)
            avg_displacement_x[row_index, col_index] += x_data[i]
            avg_displacement_y[row_index, col_index] += y_data[i]
            counts[row_index, col_index] += 1

        # Check for NaN values in the counts array
        if np.isnan(np.sum(counts)):
            print("Error: Counts array contains NaN values.")
        else:
            # Avoid division by zero
            counts[counts == 0] = 1

            # Compute average displacement vectors
            avg_displacement_x /= counts
            avg_displacement_y /= counts

            # Compute angles of displacement vectors
            angles = np.arctan2(avg_displacement_y, avg_displacement_x)
            magnitude = np.sqrt(avg_displacement_x ** 2 + avg_displacement_y ** 2)

            max_magnitude = np.max(magnitude)
            threshold = threshold_percentage / 100 * max_magnitude

            # Check for NaN values in the magnitude array
            if np.isnan(np.sum(magnitude)):
                print("Error: Magnitude array contains NaN values.")
            else:
                # Create the plot
                fig, ax1 = plt.subplots(figsize=(10, 8))

                # Plot displacement vectors in the first subplot (left)
                sns.kdeplot(x=x_data, y=y_data, cmap="viridis", fill=True, thresh=0.02, ax=ax1)
                for i in range(num_rows):
                    for j in range(num_cols):
                        x_center = x_min + (j + 0.5) * grid_width
                        y_center = y_min + (i + 0.5) * grid_height
                        avg_dx = avg_displacement_x[i, j] - x_center
                        avg_dy = avg_displacement_y[i, j] - y_center
                        avg_magnitude = magnitude[i, j]
                        # Check if there are embedding points within the sub-grid
                        subgrid_x_min = x_min + j * grid_width
                        subgrid_x_max = subgrid_x_min + grid_width
                        subgrid_y_min = y_min + i * grid_height
                        subgrid_y_max = subgrid_y_min + grid_height
                        embedding_indices = np.where(
                            (x_data >= subgrid_x_min) & (x_data <= subgrid_x_max) & (y_data >= subgrid_y_min) & (
                                    y_data <= subgrid_y_max)
                        )[0]
                        if len(embedding_indices) == 0:
                            continue  # Skip if there are no embedding points in the sub-grid
                        if avg_magnitude <= threshold:
                            ax1.plot(x_center, y_center, 'ro', markersize=5, color = 'black')  # Plot dot
                        else:
                            angle = angles[i, j]
                            color = (angle + np.pi) / (2 * np.pi)  # Normalize angle to [0, 1] for colormap
                            arrow_length = np.sqrt(avg_dx ** 2 + avg_dy ** 2)
                            ax1.arrow(x_center, y_center, avg_dx, avg_dy, head_width=0.5, head_length=0.5, fc=plt.cm.hsv(color), ec=plt.cm.hsv(color))
                        

                # Set plot labels and title for the first subplot
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')

                # Set axis limits for the first subplot
                ax1.set_xlim(x_min, x_max)
                ax1.set_ylim(y_min, y_max)

                # Remove axes for the first subplot
                ax1.axis('off')

                # Plot color wheel legend in the second subplot (right)
                ax2 = fig.add_subplot(122, projection='polar')
                ax2.grid(False)
                ax2.set_yticklabels([])                   
                ax2.tick_params(pad=10, labelsize=18)
                ax2.spines['polar'].set_visible(False)
                ax2.set_position([1, 0.4, 0.1, 0.2])  # Position and resize the subplot

                norm = mpl.colors.Normalize(0, 2*np.pi)
                n = 200  
                t = np.linspace(0,2*np.pi,n)   
                r = np.linspace(0.8,1,2)        
                rg, tg = np.meshgrid(r,t)      
                c = tg                         
                im = ax2.pcolormesh(t, r, c.T, norm=norm, cmap='hsv')
                plt.suptitle("Average Displacement Vector Field")
                plt.show()


def plot_average_heatmap_subplots(data, features, feature_names, feature_ranges, percentage=100, grid_rows=15, grid_cols=15):
    # Determine the number of features
    num_features = features.shape[1]

    # Calculate the number of rows and columns for the subplot grid
    num_rows = 4
    num_cols = 6

    # Sample a percentage of the data
    num_samples = int(percentage / 100 * data.shape[0])
    sampled_indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    sampled_data = data[sampled_indices]
    sampled_features = features[sampled_indices]

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 9))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each feature and plot it in a subplot
    for i, ax in enumerate(axes):
        if i < num_features:
            feature_values = sampled_features[:, i]

            # Example x-y data (replace this with your own data)
            x_data = sampled_data[:, 0]
            y_data = sampled_data[:, 1]

            # Define the grid dimensions
            grid_width = (np.max(x_data) - np.min(x_data)) / grid_cols
            grid_height = (np.max(y_data) - np.min(y_data)) / grid_rows

            # Initialize heatmap array to store sum of values for each grid cell
            heatmap_sum = np.zeros((grid_rows, grid_cols))
            heatmap_count = np.zeros((grid_rows, grid_cols))

            # Compute sum of values in each grid cell
            for j in range(len(x_data)):
                row_index = min(int((y_data[j] - np.min(y_data)) // grid_height), grid_rows - 1)
                col_index = min(int((x_data[j] - np.min(x_data)) // grid_width), grid_cols - 1)
                heatmap_sum[row_index, col_index] += feature_values[j]
                heatmap_count[row_index, col_index] += 1

            # Avoid division by zero
            heatmap_count[heatmap_count == 0] = 1

            # Compute average value in each grid cell
            avg_heatmap = heatmap_sum / heatmap_count

            # Plot average heatmap with inverted colors
            im = ax.imshow(avg_heatmap, cmap='jet', interpolation='kaiser')  # Use 'hot_r' to invert the colormap; interpolation bilinear, none, gaussian
            ax.set_title(feature_names[i])
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(feature_ranges[i], rotation=270, va='bottom')

            # Set background color to white
            ax.set_facecolor('white')
        else:
            ax.axis('off')  # Remove axis for any unused subplots

    plt.suptitle("Average Feature Values Heatmap Subplots")
    plt.tight_layout()

    plt.show()


def plot_3D_vector_field(embedding_data, session_ends, grid_rows=15, grid_cols=15, grid_depth=15, scale=10, width = 1000, height = 800):
    # Extract x, y, z data from the 3-dimensional PCA data
    x_data = embedding_data[:, 0]
    y_data = embedding_data[:, 1]
    z_data = embedding_data[:, 2]

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols
    num_depth = grid_depth

    # Calculate the range of the x, y, and z data
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    z_min, z_max = np.min(z_data), np.max(z_data)

    # Extend the x, y, and z limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    z_extend = 0.1 * z_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend
    z_min -= z_extend
    z_max += z_extend

    # Create mesh grid
    x_grid, y_grid, z_grid = np.meshgrid(
        np.linspace(x_min, x_max, num_cols),
        np.linspace(y_min, y_max, num_rows),
        np.linspace(z_min, z_max, num_depth),
        indexing='ij'
    )

    # Initialize arrays to store displacement vectors
    displacement_x = np.zeros_like(x_grid)
    displacement_y = np.zeros_like(y_grid)
    displacement_z = np.zeros_like(z_grid)
    counts = np.zeros_like(x_grid, dtype=int)

    # Compute displacement vectors for the average session
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Extract data for current session
        session_x_data = x_data[start_index:end_index]
        session_y_data = y_data[start_index:end_index]
        session_z_data = z_data[start_index:end_index]

        # Compute displacement vectors for current session
        for j in range(len(session_x_data) - 1):
            # Calculate grid indices for current and next points
            row_index_curr = np.clip(np.searchsorted(np.linspace(y_min, y_max, num_rows), session_y_data[j], side='right') - 1, 0, num_rows - 1)
            col_index_curr = np.clip(np.searchsorted(np.linspace(x_min, x_max, num_cols), session_x_data[j], side='right') - 1, 0, num_cols - 1)
            depth_index_curr = np.clip(np.searchsorted(np.linspace(z_min, z_max, num_depth), session_z_data[j], side='right') - 1, 0, num_depth - 1)
            row_index_next = np.clip(np.searchsorted(np.linspace(y_min, y_max, num_rows), session_y_data[j + 1], side='right') - 1, 0, num_rows - 1)
            col_index_next = np.clip(np.searchsorted(np.linspace(x_min, x_max, num_cols), session_x_data[j + 1], side='right') - 1, 0, num_cols - 1)
            depth_index_next = np.clip(np.searchsorted(np.linspace(z_min, z_max, num_depth), session_z_data[j + 1], side='right') - 1, 0, num_depth - 1)

            # Increment displacement vector between current and next points
            displacement_x[row_index_curr, col_index_curr, depth_index_curr] += session_x_data[j + 1] - session_x_data[j]
            displacement_y[row_index_curr, col_index_curr, depth_index_curr] += session_y_data[j + 1] - session_y_data[j]
            displacement_z[row_index_curr, col_index_curr, depth_index_curr] += session_z_data[j + 1] - session_z_data[j]
            counts[row_index_curr, col_index_curr, depth_index_curr] += 1

    # Check for zero counts
    counts[counts == 0] = 1

    # Compute average displacement vectors
    avg_displacement_x = displacement_x / counts
    avg_displacement_y = displacement_y / counts
    avg_displacement_z = displacement_z / counts

    # Create quiver plot
    fig = go.Figure(data=go.Cone(
        x=x_grid.flatten(),
        y=y_grid.flatten(),
        z=z_grid.flatten(),
        u=avg_displacement_x.flatten(),
        v=avg_displacement_y.flatten(),
        w=avg_displacement_z.flatten(),
        colorscale='hsv',
        sizemode="absolute",
        sizeref=scale,
        anchor='tail'
    ))

    fig.update_layout(
        width=width,  # Specify the width of the figure in pixels
        height=height,
    scene=dict(
        xaxis=dict(
            title='PC1',
            gridcolor='black',  # Set the grid color for the X-axis to black
            showbackground=False,  # Hide the background of the X-axis
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            linewidth=2,
            zerolinewidth=2
        ),
        yaxis=dict(
            title='PC2',
            gridcolor='black',  # Set the grid color for the Y-axis to black
            showbackground=False,  # Hide the background of the Y-axis
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            linewidth=2,
            zerolinewidth=2
        ),
        zaxis=dict(
            title='PC3',
            gridcolor='black',  # Set the grid color for the Z-axis to black
            showbackground=False,  # Hide the background of the Z-axis
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            linewidth=2,
            zerolinewidth=2
        ),
        bgcolor='black',  # Set the background color to white
    ))

    fig.show()


def plot_displacement_vector_field_direction_global(embedding_data, session_ends, grid_rows=15, grid_cols=15, scale=10):
    # Example x-y data (replace this with your own data)
    x_data = embedding_data[:, 0]
    y_data = embedding_data[:, 1]

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Calculate the range of the x and y data
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    # Extend the x and y limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate the grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each session
    session_avg_displacement_x = []
    session_avg_displacement_y = []

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Extract data for current session
        session_x_data = x_data[start_index:end_index]
        session_y_data = y_data[start_index:end_index]

        # Initialize arrays to store displacement vectors
        displacement_x = np.zeros((num_rows, num_cols))
        displacement_y = np.zeros((num_rows, num_cols))
        counts = np.zeros((num_rows, num_cols), dtype=int)

        # Compute displacement vectors for current session
        for j in range(len(session_x_data) - 1):  # Iterate until second last point
            # Calculate grid indices for current and next points
            row_index_curr = min(int((session_y_data[j] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_x_data[j] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_y_data[j + 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_x_data[j + 1] - x_min) // grid_width), num_cols - 1)

            # Increment displacement vector between current and next points
            displacement_x[row_index_curr, col_index_curr] += session_x_data[j + 1] - session_x_data[j]
            displacement_y[row_index_curr, col_index_curr] += session_y_data[j + 1] - session_y_data[j]
            counts[row_index_curr, col_index_curr] += 1

        # Check for zero counts
        counts[counts == 0] = 1

        # Compute average displacement vectors for current session
        avg_displacement_x = displacement_x / counts
        avg_displacement_y = displacement_y / counts

        session_avg_displacement_x.append(avg_displacement_x)
        session_avg_displacement_y.append(avg_displacement_y)

    # Compute overall average displacement vectors
    overall_avg_displacement_x = np.mean(session_avg_displacement_x, axis=0)
    overall_avg_displacement_y = np.mean(session_avg_displacement_y, axis=0)

    # Compute angle of displacement vectors
    angles = np.arctan2(overall_avg_displacement_y, overall_avg_displacement_x)

    # Normalize angles to [0, 1]
    normalized_angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))

    # Create grid coordinates
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Set background color to black
    plt.gca().set_facecolor('black')

    # Plot grid vectors with colored arrows based on angle
    plt.quiver(x_grid, y_grid, overall_avg_displacement_x, overall_avg_displacement_y, normalized_angles, cmap='hsv', scale=scale)

    plt.title("Average Direction Grid Vectors")
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.show()





def plot_displacement_vector_field_direction_unique_stims(embedding_data, session_ends, stims, grid_rows=15, grid_cols=15, scale=10):
    unique_stims = list(set(stims))
    num_unique_stims = len(unique_stims)

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Compute the range of x and y data
    x_min, x_max = np.min(embedding_data[:, 0]), np.max(embedding_data[:, 0])
    y_min, y_max = np.min(embedding_data[:, 1]), np.max(embedding_data[:, 1])

    # Extend the limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each stimulus type
    avg_displacement_x = {stim: np.zeros((num_rows, num_cols)) for stim in unique_stims}
    avg_displacement_y = {stim: np.zeros((num_rows, num_cols)) for stim in unique_stims}
    counts = {stim: np.zeros((num_rows, num_cols), dtype=int) for stim in unique_stims}

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Get stimulus type for current session
        current_stim = stims[i]

        # Filter data for current session
        session_data = embedding_data[start_index:end_index]

        # Compute displacement vectors for current session
        for j in range(len(session_data) - 1):
            row_index_curr = min(int((session_data[j, 1] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_data[j, 0] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_data[j + 1, 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_data[j + 1, 0] - x_min) // grid_width), num_cols - 1)

            avg_displacement_x[current_stim][row_index_curr, col_index_curr] += session_data[j + 1, 0] - session_data[j, 0]
            avg_displacement_y[current_stim][row_index_curr, col_index_curr] += session_data[j + 1, 1] - session_data[j, 1]
            counts[current_stim][row_index_curr, col_index_curr] += 1

    # Check for zero counts
    for stim in unique_stims:
        counts[stim][counts[stim] == 0] = 1

    # Compute average displacement vectors for each stimulus
    for stim in unique_stims:
        avg_displacement_x[stim] /= counts[stim]
        avg_displacement_y[stim] /= counts[stim]

    # Create grid coordinates for overall embedding
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Plot average displacement vectors for each stimulus with the original embedding in the background
    for stim in unique_stims:
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('black')

        # Plot original mean displacement vectors of the entire embedding in the background
        plt.quiver(x_grid, y_grid, np.mean(list(avg_displacement_x.values()), axis=0), np.mean(list(avg_displacement_y.values()), axis=0), color='grey', scale=scale)

        # Plot grid vectors with colored arrows based on angle for each stimulus
        angles = np.arctan2(avg_displacement_y[stim], avg_displacement_x[stim])
        normalized_angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))
        plt.quiver(x_grid, y_grid, avg_displacement_x[stim], avg_displacement_y[stim], normalized_angles, cmap='hsv', scale=scale)

        plt.title(f"Displacement by Stimulus: {stim}")
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.show()

def plot_displacement_vector_field_direction_unique_groups_stims(embedding_data, session_ends, stims, groups, grid_rows=15, grid_cols=15, scale=10):
    unique_stims = list(set(stims))
    num_unique_stims = len(unique_stims)
    unique_groups = list(set(groups))
    num_unique_groups = len(unique_groups)

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Compute the range of x and y data
    x_min, x_max = np.min(embedding_data[:, 0]), np.max(embedding_data[:, 0])
    y_min, y_max = np.min(embedding_data[:, 1]), np.max(embedding_data[:, 1])

    # Extend the limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each stimulus-group combination
    avg_displacement_x = {(stim, group): np.zeros((num_rows, num_cols)) for stim in unique_stims for group in unique_groups}
    avg_displacement_y = {(stim, group): np.zeros((num_rows, num_cols)) for stim in unique_stims for group in unique_groups}
    counts = {(stim, group): np.zeros((num_rows, num_cols), dtype=int) for stim in unique_stims for group in unique_groups}

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Get stimulus type and group for current session
        current_stim = stims[i]
        current_group = groups[i]

        # Filter data for current session
        session_data = embedding_data[start_index:end_index]

        # Compute displacement vectors for current session
        for j in range(len(session_data) - 1):
            row_index_curr = min(int((session_data[j, 1] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_data[j, 0] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_data[j + 1, 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_data[j + 1, 0] - x_min) // grid_width), num_cols - 1)

            avg_displacement_x[(current_stim, current_group)][row_index_curr, col_index_curr] += session_data[j + 1, 0] - session_data[j, 0]
            avg_displacement_y[(current_stim, current_group)][row_index_curr, col_index_curr] += session_data[j + 1, 1] - session_data[j, 1]
            counts[(current_stim, current_group)][row_index_curr, col_index_curr] += 1

    # Check for zero counts
    for stim, group in avg_displacement_x.keys():
        counts[(stim, group)][counts[(stim, group)] == 0] = 1

    # Compute average displacement vectors for each stimulus-group combination
    for (stim, group) in avg_displacement_x.keys():
        avg_displacement_x[(stim, group)] /= counts[(stim, group)]
        avg_displacement_y[(stim, group)] /= counts[(stim, group)]

    # Create grid coordinates for overall embedding
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Plot average displacement vectors for each stimulus-group combination with the original embedding in the background
    for stim, group in avg_displacement_x.keys():
        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('black')

        # Plot original mean displacement vectors of the entire embedding in the background
        plt.quiver(x_grid, y_grid, np.mean(list(avg_displacement_x.values()), axis=0), np.mean(list(avg_displacement_y.values()), axis=0), color='grey', scale=scale)

        # Plot grid vectors with colored arrows based on angle for each stimulus-group combination
        angles = np.arctan2(avg_displacement_y[(stim, group)], avg_displacement_x[(stim, group)])
        normalized_angles = (angles - np.min(angles)) / (np.max(angles) - np.min(angles))
        plt.quiver(x_grid, y_grid, avg_displacement_x[(stim, group)], avg_displacement_y[(stim, group)], normalized_angles, cmap='hsv', scale=scale)

        plt.title(f"Group: {group}, Stimulus: {stim}")
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.show()


def plot_displacement_vector_field_velocity_global(embedding_data, session_ends, grid_rows=15, grid_cols=15, scale=10):
    # Example x-y data (replace this with your own data)
    x_data = embedding_data[:, 0]
    y_data = embedding_data[:, 1]

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Calculate the range of the x and y data
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    # Extend the x and y limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate the grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each session
    session_avg_displacement_x = []
    session_avg_displacement_y = []

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Extract data for current session
        session_x_data = x_data[start_index:end_index]
        session_y_data = y_data[start_index:end_index]

        # Initialize arrays to store displacement vectors
        displacement_x = np.zeros((num_rows, num_cols))
        displacement_y = np.zeros((num_rows, num_cols))
        counts = np.zeros((num_rows, num_cols), dtype=int)

        # Compute displacement vectors for current session
        for j in range(len(session_x_data) - 1):  # Iterate until second last point
            # Calculate grid indices for current and next points
            row_index_curr = min(int((session_y_data[j] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_x_data[j] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_y_data[j + 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_x_data[j + 1] - x_min) // grid_width), num_cols - 1)

            # Increment displacement vector between current and next points
            displacement_x[row_index_curr, col_index_curr] += session_x_data[j + 1] - session_x_data[j]
            displacement_y[row_index_curr, col_index_curr] += session_y_data[j + 1] - session_y_data[j]
            counts[row_index_curr, col_index_curr] += 1

        # Check for zero counts
        counts[counts == 0] = 1

        # Compute average displacement vectors for current session
        avg_displacement_x = displacement_x / counts
        avg_displacement_y = displacement_y / counts

        session_avg_displacement_x.append(avg_displacement_x)
        session_avg_displacement_y.append(avg_displacement_y)

    # Compute overall average displacement vectors
    overall_avg_displacement_x = np.mean(session_avg_displacement_x, axis=0)
    overall_avg_displacement_y = np.mean(session_avg_displacement_y, axis=0)

    # Compute magnitude of displacement vectors
    magnitude = np.sqrt(overall_avg_displacement_x ** 2 + overall_avg_displacement_y ** 2)

    # Create grid coordinates
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Set background color to black
    plt.gca().set_facecolor('black')

    # Plot grid vectors with colored arrows based on magnitude
    plt.quiver(x_grid, y_grid, overall_avg_displacement_x, overall_avg_displacement_y, magnitude, cmap='hot', scale=scale)

    plt.title("Average Direction Grid Vectors")
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.show()


def plot_displacement_vector_field_velocity_unique_stims(embedding_data, session_ends, stims, grid_rows=15, grid_cols=15, scale=10):
    unique_stims = list(set(stims))
    num_unique_stims = len(unique_stims)

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Compute the range of x and y data
    x_min, x_max = np.min(embedding_data[:, 0]), np.max(embedding_data[:, 0])
    y_min, y_max = np.min(embedding_data[:, 1]), np.max(embedding_data[:, 1])

    # Extend the limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each stimulus type
    avg_displacement_x = {stim: np.zeros((num_rows, num_cols)) for stim in unique_stims}
    avg_displacement_y = {stim: np.zeros((num_rows, num_cols)) for stim in unique_stims}
    counts = {stim: np.zeros((num_rows, num_cols), dtype=int) for stim in unique_stims}

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Get stimulus type for current session
        current_stim = stims[i]

        # Filter data for current session
        session_data = embedding_data[start_index:end_index]

        # Compute displacement vectors for current session
        for j in range(len(session_data) - 1):
            row_index_curr = min(int((session_data[j, 1] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_data[j, 0] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_data[j + 1, 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_data[j + 1, 0] - x_min) // grid_width), num_cols - 1)

            avg_displacement_x[current_stim][row_index_curr, col_index_curr] += session_data[j + 1, 0] - session_data[j, 0]
            avg_displacement_y[current_stim][row_index_curr, col_index_curr] += session_data[j + 1, 1] - session_data[j, 1]
            counts[current_stim][row_index_curr, col_index_curr] += 1

    # Check for zero counts
    for stim in unique_stims:
        counts[stim][counts[stim] == 0] = 1

    # Compute average displacement vectors for each stimulus
    for stim in unique_stims:
        avg_displacement_x[stim] /= counts[stim]
        avg_displacement_y[stim] /= counts[stim]

    # Create the plot for each unique stimulus
    for stim in unique_stims:
        # Compute overall average displacement vectors
        overall_avg_displacement_x = np.mean(list(avg_displacement_x.values()), axis=0)
        overall_avg_displacement_y = np.mean(list(avg_displacement_y.values()), axis=0)

        # Compute magnitude of displacement vectors for overall embedding
        magnitude = np.sqrt(overall_avg_displacement_x ** 2 + overall_avg_displacement_y ** 2)

        # Create grid coordinates for overall embedding
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.gca().set_facecolor('black')

        # Plot original mean displacement vectors of the entire embedding in the background
        plt.quiver(x_grid, y_grid, overall_avg_displacement_x, overall_avg_displacement_y, color='grey', scale=scale)

        # Compute magnitude of displacement vectors
        magnitude = np.sqrt(avg_displacement_x[stim] ** 2 + avg_displacement_y[stim] ** 2)

        # Normalize magnitudes to [0, 1]
        normalized_magnitudes = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

        # Plot average displacement vectors for current stimulus
        plt.quiver(x_grid, y_grid, avg_displacement_x[stim], avg_displacement_y[stim], normalized_magnitudes, cmap='hot', scale=scale)

        plt.title(f"Average Magnitude Grid Vectors for Stimulus {stim}")
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.show()


def plot_displacement_vector_field_velocity_unique_groups_stims(embedding_data, session_ends, stims, groups, grid_rows=15, grid_cols=15, scale=10):
    unique_stims = list(set(stims))
    num_unique_stims = len(unique_stims)
    unique_groups = list(set(groups))
    num_unique_groups = len(unique_groups)

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Compute the range of x and y data
    x_min, x_max = np.min(embedding_data[:, 0]), np.max(embedding_data[:, 0])
    y_min, y_max = np.min(embedding_data[:, 1]), np.max(embedding_data[:, 1])

    # Extend the limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each stimulus-group combination
    avg_displacement_x = {(stim, group): np.zeros((num_rows, num_cols)) for stim in unique_stims for group in unique_groups}
    avg_displacement_y = {(stim, group): np.zeros((num_rows, num_cols)) for stim in unique_stims for group in unique_groups}
    counts = {(stim, group): np.zeros((num_rows, num_cols), dtype=int) for stim in unique_stims for group in unique_groups}

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Get stimulus type and group for current session
        current_stim = stims[i]
        current_group = groups[i]

        # Filter data for current session
        session_data = embedding_data[start_index:end_index]

        # Compute displacement vectors for current session
        for j in range(len(session_data) - 1):
            row_index_curr = min(int((session_data[j, 1] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_data[j, 0] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_data[j + 1, 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_data[j + 1, 0] - x_min) // grid_width), num_cols - 1)

            avg_displacement_x[(current_stim, current_group)][row_index_curr, col_index_curr] += session_data[j + 1, 0] - session_data[j, 0]
            avg_displacement_y[(current_stim, current_group)][row_index_curr, col_index_curr] += session_data[j + 1, 1] - session_data[j, 1]
            counts[(current_stim, current_group)][row_index_curr, col_index_curr] += 1

    # Check for zero counts
    for stim, group in avg_displacement_x.keys():
        counts[(stim, group)][counts[(stim, group)] == 0] = 1

    # Compute average displacement vectors for each stimulus-group combination
    for (stim, group) in avg_displacement_x.keys():
        avg_displacement_x[(stim, group)] /= counts[(stim, group)]
        avg_displacement_y[(stim, group)] /= counts[(stim, group)]

    # Create grid coordinates for overall embedding
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Plot average displacement vectors for each stimulus-group combination with the original embedding in the background
    for stim, group in avg_displacement_x.keys():
        # Calculate magnitude of displacement vectors
        magnitude = np.sqrt(avg_displacement_x[(stim, group)] ** 2 + avg_displacement_y[(stim, group)] ** 2)

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.gca().set_facecolor('black')

        # Plot original mean displacement vectors of the entire embedding in the background
        plt.quiver(x_grid, y_grid, np.mean(list(avg_displacement_x.values()), axis=0), np.mean(list(avg_displacement_y.values()), axis=0), color='grey', scale=scale)

        # Plot grid vectors with colored arrows based on magnitude for each stimulus-group combination
        plt.quiver(x_grid, y_grid, avg_displacement_x[(stim, group)], avg_displacement_y[(stim, group)], magnitude, cmap='hot', scale=scale)

        plt.title(f"Group: {group}, Stimulus: {stim}")
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.show()


def plot_unique_session_against_global_embedding(embedding_data, session_ends, session, grid_rows=15, grid_cols=15, embedding_scale=10, session_scale=5, overlay_color = "purple"):
    # Example x-y data (replace this with your own data)
    x_data = embedding_data[session_ends[session-1]:session_ends[session], 0]
    y_data = embedding_data[session_ends[session-1]:session_ends[session], 1]

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Calculate the range of the x and y data
    x_min, x_max = np.min(embedding_data[:, 0]), np.max(embedding_data[:, 0])
    y_min, y_max = np.min(embedding_data[:, 1]), np.max(embedding_data[:, 1])

    # Extend the x and y limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate the grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each session
    session_avg_displacement_x = []
    session_avg_displacement_y = []

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Extract data for current session
        session_x_data = embedding_data[start_index:end_index, 0]
        session_y_data = embedding_data[start_index:end_index, 1]

        # Initialize arrays to store displacement vectors
        displacement_x = np.zeros((num_rows, num_cols))
        displacement_y = np.zeros((num_rows, num_cols))
        counts = np.zeros((num_rows, num_cols), dtype=int)

        # Compute displacement vectors for current session
        for j in range(len(session_x_data) - 1):  # Iterate until second last point
            # Calculate grid indices for current and next points
            row_index_curr = min(int((session_y_data[j] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_x_data[j] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_y_data[j + 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_x_data[j + 1] - x_min) // grid_width), num_cols - 1)

            # Increment displacement vector between current and next points
            displacement_x[row_index_curr, col_index_curr] += session_x_data[j + 1] - session_x_data[j]
            displacement_y[row_index_curr, col_index_curr] += session_y_data[j + 1] - session_y_data[j]
            counts[row_index_curr, col_index_curr] += 1

        # Check for zero counts
        counts[counts == 0] = 1

        # Compute average displacement vectors for current session
        avg_displacement_x = displacement_x / counts
        avg_displacement_y = displacement_y / counts

        session_avg_displacement_x.append(avg_displacement_x)
        session_avg_displacement_y.append(avg_displacement_y)

    # Compute overall average displacement vectors
    overall_avg_displacement_x = np.mean(session_avg_displacement_x, axis=0)
    overall_avg_displacement_y = np.mean(session_avg_displacement_y, axis=0)

    # Compute magnitude of displacement vectors for overall embedding
    magnitude = np.sqrt(overall_avg_displacement_x ** 2 + overall_avg_displacement_y ** 2)

    # Create grid coordinates for overall embedding
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Create the plot
    plt.figure(figsize=(8, 8))

    # Set background color to black
    plt.gca().set_facecolor('black')

    # Plot grid vectors with colored arrows based on magnitude for overall embedding
    plt.quiver(x_grid, y_grid, overall_avg_displacement_x, overall_avg_displacement_y, magnitude, cmap='gray', scale=embedding_scale)

    # Compute displacement vectors for the particular session using the same grid as the overall embedding
    session_displacement_x = np.diff(x_data)
    session_displacement_y = np.diff(y_data)

    # Initialize arrays to store displacement vectors for the session
    session_avg_displacement_x = np.zeros_like(x_grid)
    session_avg_displacement_y = np.zeros_like(y_grid)

    # Compute displacement vectors for the session
    for j in range(len(session_displacement_x)):
        # Calculate grid indices for current and next points
        row_index_curr = min(int((y_data[j] - y_min) // grid_height), num_rows - 1)
        col_index_curr = min(int((x_data[j] - x_min) // grid_width), num_cols - 1)
        row_index_next = min(int((y_data[j + 1] - y_min) // grid_height), num_rows - 1)
        col_index_next = min(int((x_data[j + 1] - x_min) // grid_width), num_cols - 1)

        # Increment displacement vector between current and next points
        session_avg_displacement_x[row_index_curr, col_index_curr] += session_displacement_x[j]
        session_avg_displacement_y[row_index_curr, col_index_curr] += session_displacement_y[j]

    # Compute magnitude of displacement vectors for the session
    session_magnitude = np.sqrt(session_avg_displacement_x ** 2 + session_avg_displacement_y ** 2)

    # Normalize displacement vectors to get direction
    session_direction_x = session_avg_displacement_x / session_magnitude
    session_direction_y = session_avg_displacement_y / session_magnitude

    # Plot arrows for the particular session based on the overall embedding grid
    plt.quiver(x_grid, y_grid, session_direction_x, session_direction_y, color='pink', scale=session_scale, linewidth = 2, edgecolor = overlay_color)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.show()


def plot_curl_streamlines_global(embedding_data, session_ends, grid_rows=15, grid_cols=15, scale=10):
    # Example x-y data (replace this with your own data)
    x_data = embedding_data[:, 0]
    y_data = embedding_data[:, 1]

    # Define the grid dimensions
    num_rows = grid_rows
    num_cols = grid_cols

    # Calculate the range of the x and y data
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)

    # Extend the x and y limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_extend = 0.1 * x_range
    y_extend = 0.1 * y_range
    x_min -= x_extend
    x_max += x_extend
    y_min -= y_extend
    y_max += y_extend

    # Calculate the grid cell size based on the extended bounds
    grid_width = (x_max - x_min) / num_cols
    grid_height = (y_max - y_min) / num_rows

    # Initialize arrays to store average displacement vectors for each session
    session_avg_displacement_x = []
    session_avg_displacement_y = []

    # Iterate over sessions
    for i in range(len(session_ends)):
        # Determine start and end indices for current session
        start_index = 0 if i == 0 else session_ends[i - 1]
        end_index = session_ends[i]

        # Extract data for current session
        session_x_data = x_data[start_index:end_index]
        session_y_data = y_data[start_index:end_index]

        # Initialize arrays to store displacement vectors
        displacement_x = np.zeros((num_rows, num_cols))
        displacement_y = np.zeros((num_rows, num_cols))
        counts = np.zeros((num_rows, num_cols), dtype=int)

        # Compute displacement vectors for current session
        for j in range(len(session_x_data) - 1):  # Iterate until second last point
            # Calculate grid indices for current and next points
            row_index_curr = min(int((session_y_data[j] - y_min) // grid_height), num_rows - 1)
            col_index_curr = min(int((session_x_data[j] - x_min) // grid_width), num_cols - 1)
            row_index_next = min(int((session_y_data[j + 1] - y_min) // grid_height), num_rows - 1)
            col_index_next = min(int((session_x_data[j + 1] - x_min) // grid_width), num_cols - 1)

            # Increment displacement vector between current and next points
            displacement_x[row_index_curr, col_index_curr] += session_x_data[j + 1] - session_x_data[j]
            displacement_y[row_index_curr, col_index_curr] += session_y_data[j + 1] - session_y_data[j]
            counts[row_index_curr, col_index_curr] += 1

        # Check for zero counts
        counts[counts == 0] = 1

        # Compute average displacement vectors for current session
        avg_displacement_x = displacement_x / counts
        avg_displacement_y = displacement_y / counts

        session_avg_displacement_x.append(avg_displacement_x)
        session_avg_displacement_y.append(avg_displacement_y)

    # Compute overall average displacement vectors
    overall_avg_displacement_x = np.mean(session_avg_displacement_x, axis=0)
    overall_avg_displacement_y = np.mean(session_avg_displacement_y, axis=0)

    # Compute curl
    curl = np.gradient(overall_avg_displacement_y, axis=1) - np.gradient(overall_avg_displacement_x, axis=0)

    # Create grid coordinates
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, num_cols), np.linspace(y_min, y_max, num_rows))

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Set background color to black
    plt.gca().set_facecolor('black')

    # Plot curl
    plt.imshow(curl, cmap='viridis', extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=0.7)
    plt.colorbar(label='Curl')

    # Plot streamgrid
    plt.streamplot(x_grid, y_grid, overall_avg_displacement_x, overall_avg_displacement_y, color='black', linewidth=1, arrowsize=1.5)
    plt.show()