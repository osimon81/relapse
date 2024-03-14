from utils.preprocessing_functions import *

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


def pca_transform(data, n_components):
    """
    Perform PCA transformation on a 2-dimensional array of data (features) with a progress bar.

    Parameters:
        features (array-like): The input features to be transformed.
        n_components (int): The number of principal components to retain.

    Returns:
        array-like: Transformed features.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)

    return pca_result



def fit_transform_with_progress(model, data):
    transformed_data = []
    total_samples = len(data)
    for i in tqdm(range(0, total_samples, 1000), desc="Fitting t-SNE", total=total_samples // 1000):
        transformed_data.append(model.fit_transform(data[i:i+1000]))
    return np.vstack(transformed_data)

# Define a wrapper function to fit and transform with tqdm
def fit_transform_with_progress(model, data):
    transformed_data = []
    total_samples = len(data)
    for i in tqdm(range(0, total_samples, 1000), desc="Fitting model", total=total_samples // 1000):
        transformed_data.append(model.fit_transform(data[i:i+1000]))
    return np.vstack(transformed_data)


def train_variational_autoencoder(paw_withdrawal_features, sliding_window_size, epochs_n, batch_size, validation_split):
    # Create sliding windows for each sample in the data
    data_windows = create_sliding_windows(paw_withdrawal_features, sliding_window_size)

    # Define the autoencoder architecture
    input_data = Input(shape=(sliding_window_size, paw_withdrawal_features.shape[1]))

    # Encoder
    encoded = Dense(8, activation='relu')(input_data)
    encoded = Dense(4, activation='relu')(encoded)

    # Decoder
    decoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(paw_withdrawal_features.shape[1], activation='sigmoid')(decoded)

    # Create autoencoder model
    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')  # Mean squared error loss for reconstruction

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(data_windows, test_size=validation_split, random_state=42)

    # Train the autoencoder
    autoencoder.fit(train_data, train_data, epochs=epochs_n, batch_size=batch_size, validation_data=(val_data, val_data))

    # Extract the encoder part of the autoencoder
    encoder = Model(input_data, encoded)

    # Encode the input data
    encoded_data = encoder.predict(data_windows)

    # Reshape the encoded data for PCA visualization
    encoded_data_flat = encoded_data.reshape(encoded_data.shape[0], -1)

    return encoded_data_flat
