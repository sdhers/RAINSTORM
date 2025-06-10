# rainstorm/model_evaluation.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import Dict, List
from pathlib import Path
import logging
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Model Evaluation Functions

def evaluate(y_pred: np.ndarray, y_true: np.ndarray, show_report: bool = False) -> Dict[str, float]:
    """
    Evaluates model performance using various metrics.

    Args:
        y_pred (np.ndarray): Predicted values (probabilities or continuous).
        y_true (np.ndarray): True labels.
        show_report (bool): If True, prints a detailed classification report.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    # Ensure y_pred and y_true are NumPy arrays and then flatten
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.to_numpy()
    y_pred = y_pred.flatten()

    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.to_numpy()
    y_true = y_true.flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # For classification metrics, binarize predictions and true labels
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true_binary = (y_true > 0.5).astype(int)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Handle cases where precision/recall/f1 might be undefined due to no positive samples
    # or no predicted positive samples. `zero_division=0` sets the score to 0 in such cases.
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
    }

    if show_report:
        print("\n--- Classification Report ---")
        print(classification_report(y_true_binary, y_pred_binary, zero_division=0))
        print("-----------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("-----------------------------\n")

    return metrics

def build_evaluation_dict(modeling_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Creates a dictionary to evaluate the performance of the models.

    Args:
        modeling_path (Path): Path to modeling.yaml with colabel settings.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping model names to their evaluation dataframes.
    """
    # Load parameters
    modeling = load_yaml(modeling_path)
    colabels = modeling.get("colabels", {})
    colabels_path = colabels.get("colabels_path")
    labelers = colabels.get("labelers", [])

    # Open the colabels file
    colabels_df = pd.read_csv(colabels_path)
    position = colabels_df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy() # Extract positions, excluding tail-related columns
    manual_labels = pd.concat([colabels_df.filter(regex=name).copy() for name in labelers], axis=1) # Extract individual labelers' columns
    geometric = colabels_df.filter(regex='Geometric').copy() # Extract geometric labels

    # Create a dictionary to store evaluation results
    evaluation_dict = {}
    evaluation_dict['position'] = position
    # add the manual labels to the dictionary
    for name in labelers:
        evaluation_dict[name] = colabels_df.filter(regex=name).copy()
    evaluation_dict['manual_labels'] = manual_labels
    evaluation_dict['geometric'] = geometric

    return evaluation_dict


def create_chimera_and_loo_mean(df: pd.DataFrame, seed: int = None) -> Dict[str, np.ndarray]:
    """
    Creates a chimera DataFrame by randomly selecting columns for each row.

    Args:
        df (pd.DataFrame): DataFrame to create chimera from.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'chimera' and 'LOO_mean' arrays.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    
    if df.shape[1] == 1:
        # If only one column, chimera and loo_mean are the same as the input
        chimera = df.copy()
        chimera.columns = ['chimera']
        loo_mean = df.copy()
        loo_mean.columns = ['loo_mean']
        return chimera, loo_mean
    
    n_cols = df.shape[1]
    
    # Randomly select a column index (0 to n_cols) for each row
    chosen_indices = np.random.randint(0, n_cols, size=len(df))
    
    # Use numpy to get the values of the randomly chosen columns
    chimera_values = df.values[np.arange(len(df)), chosen_indices]
    
    # Calculate the sum of all columns for each row
    row_sums = df.sum(axis=1)
    
    # Subtract the chosen values from the row sums and divide by (n_cols - 1) to get the mean
    remaining_means = (row_sums - chimera_values) / (n_cols - 1)
    
    # Assign the new columns to the DataFrame
    chimera = pd.DataFrame(chimera_values, columns=['chimera'])
    loo_mean = pd.DataFrame(remaining_means, columns=['loo_mean'])

    return {"chimera": chimera, "loo_mean": loo_mean}


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

