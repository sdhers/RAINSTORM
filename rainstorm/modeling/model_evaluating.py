# rainstorm/model_evaluation.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import tensorflow as tf

# Import necessary utilities from the main utils file
from .utils import evaluate

logger = logging.getLogger(__name__)

# %% Model Evaluation Functions

def build_evaluation_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Builds a dictionary of evaluation metrics (MSE, MAE, R2, Accuracy, Precision, Recall, F1).

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels (probabilities or continuous values).

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    return evaluate(y_pred, y_true, show_report=False)


def create_chimera_and_loo_mean(model_outputs: Dict[str, np.ndarray], labeler_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Creates 'chimera' (mean of models) and 'LOO_mean' (mean of labelers) arrays.

    Args:
        model_outputs (Dict[str, np.ndarray]): Dictionary of model predictions.
        labeler_outputs (Dict[str, np.ndarray]): Dictionary of labeler data.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'chimera' and 'LOO_mean' arrays.
    """
    num_models = len(model_outputs)
    num_labelers = len(labeler_outputs)

    # Chimera: Mean of all model outputs
    if num_models > 0:
        chimera = np.mean([arr for arr in model_outputs.values()], axis=0)
    else:
        chimera = np.array([])
        logger.warning("No model outputs provided for chimera calculation.")

    # LOO_mean: Mean of all labelers (Leave-One-Out style, meaning all labelers averaged)
    # The term LOO_mean here refers to the overall mean of labelers, not an actual LOO iteration.
    if num_labelers > 0:
        loo_mean = np.mean([arr for arr in labeler_outputs.values()], axis=0)
    else:
        loo_mean = np.array([])
        logger.warning("No labeler outputs provided for LOO_mean calculation.")

    return {"chimera": chimera, "LOO_mean": loo_mean}


def calculate_cosine_sim(data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Calculates cosine similarity between all pairs of arrays in the input dictionary.

    Args:
        data_dict (Dict[str, np.ndarray]): Dictionary of arrays (e.g., model outputs, labeler data).

    Returns:
        pd.DataFrame: DataFrame containing the cosine similarity matrix.
    """
    keys = list(data_dict.keys())
    num_items = len(keys)
    
    if num_items < 2:
        logger.warning("Need at least two items to calculate cosine similarity.")
        return pd.DataFrame()

    similarity_matrix = np.zeros((num_items, num_items))

    for i in range(num_items):
        for j in range(num_items):
            # Reshape to 2D array for cosine_similarity function (each sample is a row)
            vec1 = data_dict[keys[i]].reshape(1, -1)
            vec2 = data_dict[keys[j]].reshape(1, -1)
            
            # cosine_similarity returns a 2D array [[similarity_score]]
            similarity_matrix[i, j] = cosine_similarity(vec1, vec2)[0, 0]

    cosine_df = pd.DataFrame(similarity_matrix, index=keys, columns=keys)
    return cosine_df


def plot_PCA(data_dict: Dict[str, np.ndarray], n_components: int = 2) -> None:
    """
    Performs PCA on the input data and plots the results.

    Args:
        data_dict (Dict[str, np.ndarray]): Dictionary of arrays (e.g., model outputs, labeler data).
        n_components (int): Number of PCA components to calculate.
    """
    if len(data_dict) < 2:
        logger.warning("Not enough data series to perform PCA and plot meaningfully (need at least 2).")
        return

    # Prepare data: each array becomes a row in the input for PCA
    keys = list(data_dict.keys())
    data_for_pca = np.array([arr for arr in data_dict.values()])

    # Ensure data is 2D: (n_samples, n_features)
    if data_for_pca.ndim == 1: # If only one sample or all are 1D lists
        data_for_pca = data_for_pca.reshape(1, -1)
    elif data_for_pca.ndim > 2: # If arrays are 2D or more, flatten them
        data_for_pca = data_for_pca.reshape(data_for_pca.shape[0], -1)

    if data_for_pca.shape[1] < n_components:
        logger.warning(f"Number of features ({data_for_pca.shape[1]}) is less than n_components ({n_components}). Adjusting n_components.")
        n_components = data_for_pca.shape[1]
        if n_components < 1:
            logger.error("Cannot perform PCA with less than 1 component.")
            return

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_for_pca)

    explained_variance = pca.explained_variance_ratio_ * 100
    logger.info(f"Explained variance by {n_components} components: {explained_variance.sum():.2f}%")

    if n_components == 2:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=keys)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=pca_df.index, s=100, style=pca_df.index)
        
        # Annotate points
        for i, txt in enumerate(keys):
            plt.annotate(txt, (pca_df['PC1'][i], pca_df['PC2'][i]), textcoords="offset points", xytext=(5,5), ha='center')

        plt.title('PCA of Model and Labeler Outputs')
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    elif n_components == 3:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'], index=keys)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], s=100)
        
        for i, txt in enumerate(keys):
            ax.text(pca_df['PC1'][i], pca_df['PC2'][i], pca_df['PC3'][i], txt, fontsize=9)

        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
        ax.set_zlabel(f'Principal Component 3 ({explained_variance[2]:.2f}%)')
        ax.set_title('3D PCA of Model and Labeler Outputs')
        plt.show()
    else:
        logger.info(f"PCA with {n_components} components calculated. For visualization, n_components should be 2 or 3.")


def plot_history(history: tf.keras.callbacks.History, model_name: str) -> None:
    """
    Plots the training and validation accuracy and loss from a Keras history object.

    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
        model_name (str): Name of the model for plot titles.
    """
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], 'r', label='Training loss')
    plt.plot(epochs, hist['val_loss'], 'b', label='Validation loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['accuracy'], 'r', label='Training accuracy')
    plt.plot(epochs, hist['val_accuracy'], 'b', label='Validation accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_performance_on_video(
    folder_path: Path,
    model_paths: Dict[str, Path],
    labelers: Dict[str, Path],
    fps: int = 25,
    bodyparts: List[str] = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'],
    targets: List[str] = ['obj_1', 'obj_2'],
    plot_tgt: str = 'obj_1'
) -> go.Figure:
    """
    Plots the performance of models against human labelers for a specific video/target.

    Args:
        folder_path (Path): Path to the folder containing position files and labeler data.
                            Assumes `positions` and labeler subdirectories exist.
        model_paths (Dict[str, Path]): Dictionary mapping model names to their saved model file paths.
        labelers (Dict[str, Path]): Dictionary mapping labeler names to their label CSV file paths
                                    (e.g., 'Example_Marian.csv').
        fps (int): Frames per second of the video.
        bodyparts (List[str]): List of body parts used in the position data.
        targets (List[str]): List of target object names.
        plot_tgt (str): The specific target column name to plot (e.g., 'obj_1').

    Returns:
        go.Figure: A Plotly figure object displaying the performance.
    """
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder path not found: {folder_path}")

    # Load labeler outputs
    labeler_outputs = {}
    for labeler_name, labeler_file in labelers.items():
        labeler_filepath = folder_path / labeler_file # Assuming labeler_file is relative to folder_path
        if not labeler_filepath.is_file():
            logger.warning(f"Labeler file not found: {labeler_filepath}. Skipping {labeler_name}.")
            continue
        df_label = pd.read_csv(labeler_filepath)
        if plot_tgt not in df_label.columns:
            logger.warning(f"Target '{plot_tgt}' not found in labeler file {labeler_filepath}. Skipping {labeler_name}.")
            continue
        labeler_outputs[labeler_name] = df_label[plot_tgt]

    if not labeler_outputs:
        logger.error("No valid labeler files found or target column missing. Cannot plot performance.")
        return go.Figure()

    # Load position data for models
    # This requires knowing which position file corresponds to the labeler files.
    # A robust solution would involve matching filenames or passing the specific position file.
    # For now, let's assume the labeler_file base name matches a position file base name.
    # Take the first labeler file as a reference to find the corresponding position file.
    first_labeler_filename = next(iter(labelers.values()))
    # Assuming labeler file is 'Example_Marian.csv' and position file is 'Example_Marian_position.csv'
    # Or, if the position file is more general, like 'video_X_position.csv'
    # We need a more robust way to link labeler files to a specific position file.
    # For this example, let's assume `folder_path` contains the actual position file
    # and the labeler files are just the labels for *that* video.
    
    # IMPROVEMENT: The `plot_performance_on_video` function in the original
    # notebook is designed to take a *single* example_path, implying a single video's data.
    # Let's adjust this to take `position_filepath` as a direct argument.
    
    # Since the original notebook was using `example_path` to load the position,
    # let's find the position file associated with the first labeler.
    
    # This needs clarification in the original notebook's context.
    # For now, let's assume `folder_path` *is* the path to a specific video's data.
    # And we need to derive the position file name from one of the labeler files.
    
    # A safer approach is to pass the explicit position file path to this function.
    # As per 3a-Create_models.ipynb: example_path refers to a directory containing "positions"
    # and labeler files. This means we need to find the specific position file that matches
    # the label files being used.
    
    # Let's infer position filename from a labeler filename.
    # If a labeler file is 'video1_labels.csv', position file might be 'video1_position.csv'.
    # This is a heuristic and might need to be more explicit.
    
    # For demonstration, let's assume a generic `position_data.csv` in `folder_path/positions`.
    # This might need to be made dynamic if each labeler file corresponds to a unique video.
    # Based on the example in 3a-Create_models.ipynb where `example_path` was a directory,
    # and `labelers_example` had `Example_Marian.csv`, the corresponding position file
    # would be something like `positions/Example_Marian_position.csv`.

    # Let's assume the `example_path` from the notebook is passed as `folder_path`
    # and we need to find a single, representative position file.
    # This is a bit of a hack, but matches the notebook's implied usage.
    
    position_data_dir = folder_path / "positions"
    position_file_base_name = Path(next(iter(labelers.values()))).stem.replace('_labels', '_position')
    position_filepath = position_data_dir / f"{position_file_base_name}.csv"

    if not position_filepath.is_file():
        # Fallback if the direct mapping doesn't work, maybe it's just a general position file
        # or needs to be specified more clearly.
        # For now, if the specific one isn't found, raise an error to indicate problem.
        raise FileNotFoundError(f"Corresponding position file not found for plotting: {position_filepath}. "
                                "Ensure the position file naming convention matches labeler files or specify it directly.")
    
    position_df = pd.read_csv(position_filepath)

    # Run models on the full position data
    model_outputs = build_and_run_models(
        modeling_path=folder_path / "modeling.yaml", # Assuming modeling.yaml is in folder_path
        model_paths=model_paths,
        position_df=position_df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy() # Ensure only relevant position columns are passed
    )

    # Create time axis
    # Use the length of one of the labeler outputs to determine total frames
    num_frames = len(next(iter(labeler_outputs.values())))
    time = np.arange(num_frames) / fps

    # Create a figure
    fig = go.Figure()

    # Add traces for labelers
    # Apply a small offset for better visibility of overlapping binary labels
    for idx, (labeler_name, labeler_data) in enumerate(labeler_outputs.items()):
        offset = 1 + 0.025 * (idx + 1) # Incremental offset for visualization
        fig.add_trace(
            go.Scatter(
                x=time,
                y=[x * offset for x in labeler_data], # Scale by offset
                mode='markers', # Markers for discrete labels
                name=labeler_name,
                marker=dict(color=f"hsl({idx * 60}, 70%, 50%)", size=5)
            )
        )

    # Add traces for models
    for model_name, model_output in model_outputs.items():
        # Ensure model output is 1D and matches length of labeler data
        if len(model_output) != num_frames:
            logger.warning(f"Model '{model_name}' output length ({len(model_output)}) does not match labeler length ({num_frames}). Cropping/padding may be needed.")
            # For plotting, clip to the minimum length to avoid errors
            model_output = model_output[:num_frames]
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=model_output,
                mode='lines', # Lines for continuous model predictions
                name=model_name,
                line=dict(width=2)
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Model and Labeler Performance on Video for Target: {plot_tgt}",
        xaxis_title="Time (s)",
        yaxis_title="Activity / Label Offset",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Legend",
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    fig.show()
    return fig

