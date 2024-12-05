# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 3b-Evaluate_models.ipynb

# %% imports

import os
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import signal
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from .utils import broaden, recenter, reshape, evaluate

# %% functions

def create_chimera_and_loo_mean(df: pd.DataFrame, seed: int = None) -> tuple:
    """Creates a chimera DataFrame by randomly selecting columns for each row.

    Args:
        df (pd.DataFrame): DataFrame to create chimera from.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the chimera DataFrame and the loo_mean DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)

    if df.empty or df.shape[1] < 2:
        raise ValueError("Input DataFrame must have at least two columns.")

    n_cols = df.shape[1]

    # Randomly select a column index (0 to n_cols) for each row
    chosen_indices = np.random.randint(0, n_cols, size=len(df))

    # Use numpy to get the values of the randomly chosen columns
    chimera_values = df.values[np.arange(len(df)), chosen_indices]

    # Calculate the sum of the first `cols_to_use` columns for each row
    row_sums = df.iloc[:, :n_cols].sum(axis=1)

    # Subtract the chosen values from the row sums and divide by (cols_to_use - 1) to get the mean
    remaining_means = (row_sums - chimera_values) / (n_cols - 1)

    # Assign the new columns to the DataFrame
    chimera = pd.DataFrame(chimera_values, columns=['chimera'])
    loo_mean = pd.DataFrame(remaining_means, columns=['loo_mean'])

    return chimera, loo_mean

def smooth_columns(df: pd.DataFrame, columns: list = [], kernel_size: int = 3, gauss_std: float = 0.6) -> pd.DataFrame:
    """Applies smoothing to a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame to apply smoothing to.
        columns (list, optional): List of columns to apply smoothing to. Defaults to [].
        kernel_size (int, optional): Size of the smoothing kernel. Defaults to 3.
        gauss_std (float, optional): Standard deviation of the Gaussian kernel. Defaults to 0.6.

    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """

    if not columns:
        columns = df.columns

    for column in columns:

        # Apply median filter
        df['med_filt'] = signal.medfilt(df[column], kernel_size=kernel_size)
        
        # Gaussian kernel
        gauss_kernel = signal.windows.gaussian(kernel_size, gauss_std)
        gauss_kernel = gauss_kernel / sum(gauss_kernel)  # Normalize kernel
        
        # Pad the median filtered data to mitigate edge effects
        pad_width = (len(gauss_kernel) - 1) // 2
        padded = np.pad(df['med_filt'], pad_width, mode='edge')
        
        # Apply convolution
        smooth = signal.convolve(padded, gauss_kernel, mode='valid')
        
        # Trim the padded edges to restore original length
        df['smooth'] = smooth[:len(df[column])]
        
        # Apply sigmoid transformation
        df[column] = round(1 / (1 + np.exp(-12*(df['smooth'] - 0.5))), 2)
        
    return df.drop(columns=['med_filt', 'smooth'])

def use_model(position, model, objects = ['obj_1', 'obj_2'], bodyparts = ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body'], recentering = True, reshaping = False, past: int = 3, future: int = 3, broad: float = 1.7):
    
    if recentering:
        position = pd.concat([recenter(position, obj, bodyparts) for obj in objects], ignore_index=True)

    if reshaping:
        position = np.array(reshape(position, past, future, broad))
    
    pred = model.predict(position) # Use the model to predict the labels
    pred = pred.flatten()
    pred = pd.DataFrame(pred, columns=['predictions'])

    n_objects = len(objects)

    # Calculate the length of each fragment
    fragment_length = len(pred) // n_objects

    # Create a list to hold each fragment
    fragments = [pred.iloc[i*fragment_length:(i+1)*fragment_length].reset_index(drop=True) for i in range(n_objects)]

    # Concatenate fragments along columns
    labels = pd.concat(fragments, axis=1)

    # Rename columns
    labels.columns = [f'{obj}' for obj in objects]

    labels = round(labels, 2)
    
    return labels

def calculate_cosine_sim(data, show_plot = True):

    # Combine all columns into a single DataFrame
    all_labelers_matrix = pd.DataFrame({name: df.iloc[:, 0] for name, df in data.items()})

    cosine_sim = pd.DataFrame(cosine_similarity(all_labelers_matrix.T), index=all_labelers_matrix.columns, columns=all_labelers_matrix.columns)
    if show_plot:
        plt.figure()
        sns.heatmap(cosine_sim.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Cosine Similarity")
        plt.show()

    return cosine_sim

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_PCA(data, make_discrete = False):
    
    # Combine all columns into a single DataFrame
    all_labelers_matrix = pd.DataFrame({name: df.iloc[:, 0] for name, df in data.items()})

    if make_discrete:
            all_labelers_matrix = (all_labelers_matrix > 0.5).astype(int)

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(all_labelers_matrix.T)

    # Plot
    plt.figure()
    for i, label in enumerate(all_labelers.keys()):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], label=label)
        plt.text(reduced_data[i, 0] + 0.1, reduced_data[i, 1] + 0.1, label, fontsize=10)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Visualization of Labeler Similarity')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()