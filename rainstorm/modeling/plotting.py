import logging
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict
from pathlib import Path

from .aux_functions import use_model
from ..geometric_classes import Point, Vector
from ..utils import configure_logging, load_yaml
configure_logging()
logger = logging.getLogger(__name__)

def plot_example_data(X: np.ndarray, y: np.ndarray, *,
                      event_label_threshold: float = 0.5,
                      position_label: str = 'Nose distance to target (cm)',
                      position_range: tuple = (-2, 25)) -> None:
    """
    Plots an example trial showing the target distance over time, highlighting periods of exploration.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features). Position should be in first two columns.
        y (np.ndarray): Binary or continuous labels for exploration (e.g., 1 for exploring, 0 otherwise).
        event_label_threshold (float): Threshold to binarize y for exploration detection.
        position_label (str): Label for the y-axis.
        position_range (tuple): Y-axis range for the plot.

    Returns:
        None. Displays an interactive Plotly figure.
    """
    # Calculate radial distance to target
    position = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

    # Threshold labels to create binary exploration indicators
    exploration = pd.DataFrame((y >= event_label_threshold).astype(int), columns=['exploration'])

    # Create time index
    time = np.arange(len(position))

    # Create base plot
    fig = go.Figure()

    # Add nose-target position trace
    fig.add_trace(go.Scatter(
        x=time,
        y=position,
        mode='lines',
        name='Position',
        line=dict(color='blue')
    ))

    # Detect changes and assign exploration event IDs
    exploration['change'] = exploration['exploration'].diff()
    exploration['event_id'] = (exploration['change'] == 1).cumsum()
    events = exploration[exploration['exploration'] == 1]

    # Add rectangles for each continuous exploration event
    for event_id, group in events.groupby('event_id'):
        if not group.empty:
            start_idx = group.index[0]
            end_idx = group.index[-1] + 1  # Inclusive range

            fig.add_shape(
                type='rect',
                x0=time[start_idx],
                x1=time[min(end_idx, len(time) - 1)],
                y0=position_range[0],
                y1=position_range[1],
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(width=0.4),
                layer='below'
            )

    # Add target distance reference line
    fig.add_hline(
        y=0,
        line=dict(color='black', dash='dash'),
        annotation_text='Target position',
        annotation_position='bottom left'
    )

    # Final layout tuning
    fig.update_layout(
        title='Exploration Events Visualization',
        xaxis_title='Frames',
        yaxis_title=position_label,
        yaxis=dict(range=position_range),
        legend=dict(
            orientation='h',
            x=0.5,
            y=1.05,
            xanchor='center',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.6)'
        )
    )

    fig.show()

def plot_history(history: tf.keras.callbacks.History, model_name: str) -> None:
    """
    Plots the training and validation accuracy and loss from a Keras history object.

    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
        model_name (str): Name of the model for plot titles.
    """
    hist = history.history

    # Create a plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=hist['loss'], 
                             mode='lines', name='Training loss'))
    fig.add_trace(go.Scatter(y=hist['val_loss'], 
                             mode='lines', name='Validation loss'))
    fig.add_trace(go.Scatter(y=hist['accuracy'], 
                             mode='lines', name='Training accuracy'))
    fig.add_trace(go.Scatter(y=hist['val_accuracy'], 
                             mode='lines', name='Validation accuracy'))

    fig.update_layout(
        title=f'Training of model {model_name}',
        xaxis_title='Epochs',
        yaxis_title='%',
        template='plotly_white',  # Optional: makes the plot cleaner
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h", 
                    bgcolor='rgba(255,255,255,0.5)')
        )

    fig.show()

def plot_lr_schedule(history: tf.keras.callbacks.History):
    """
    Plots the learning rate schedule during training based on the history object using Plotly.

    Args:
        history (tf.keras.callbacks.History): The history object returned by model.fit().
                                            It should contain 'lr' in its history.
    """
    if 'lr' not in history.history:
        logger.warning("Learning rate (lr) not found in history. Ensure LearningRateScheduler verbose is set to 1.")
        print("Warning: Learning rate (lr) not found in history. Ensure LearningRateScheduler verbose is set to 1.")
        return

    epochs = history.epoch
    lrs = history.history['lr']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=lrs, mode='lines+markers', name='Learning Rate',
                             marker=dict(size=4), line=dict(color='dodgerblue')))

    fig.update_layout(
        title='Learning Rate Schedule During Training',
        xaxis_title='Epoch',
        yaxis_title='Learning Rate (log scale)',
        yaxis_type='log', # Use log scale for LR to better visualize changes
        hovermode='x unified', # Shows hover details for all traces at a given x-position
        font=dict(
            family="Inter, sans-serif", # Using "Inter" font as specified
            size=12,
            color="#333"
        ),
        plot_bgcolor='#F8F8F8', # Light grey background for the plotting area
        paper_bgcolor='#EEEEEE', # Slightly darker grey background for the entire paper area
        margin=dict(l=40, r=40, t=80, b=40) # Adjust margins
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    fig.show()

# %% Evaluation

def plot_cosine_sim(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Calculates cosine similarity between columns and optionally displays a heatmap using Plotly.

    Args:
        data (dict): A dictionary where keys are names and values are pandas DataFrames,
                     each with at least one column.
        show_plot (bool): If True, displays an interactive Plotly heatmap.
    """
    # Combine all columns into a single DataFrame
    matrix = pd.DataFrame({name: df.iloc[:, 0] for name, df in data.items()})
    cosine_sim = pd.DataFrame(cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns)

    fig = px.imshow(cosine_sim,
                    labels=dict(x="Labeler", y="Labeler", color="Cosine Similarity"),
                    x=cosine_sim.columns,
                    y=cosine_sim.columns,
                    color_continuous_scale='RdBu_r',
                    title="Cosine Similarity",
                    # Adjust figure size for better readability
                    width=700,  # Increase width
                    height=600  # Increase height
                    )
    fig.update_xaxes(side="top")
    
    # Adjust font size for annotations (numbers inside cells)
    fig.update_traces(texttemplate="%{z:.2f}", textfont=dict(size=10)) # Ensure text is visible and formatted
    
    # Adjust margin to prevent labels from being cut off
    fig.update_layout(margin=dict(l=100, r=100, t=100, b=100)) # Increase left/right/top/bottom margins

    fig.show()

def plot_PCA(data: Dict[str, pd.DataFrame], make_discrete=False) -> go.Figure:
    """
    Performs PCA and visualizes the results in 2D using Plotly with improved aesthetics,
    distinguishing between Labelers, Models, and Methods.

    Args:
        data (dict): A dictionary where keys are names and values are pandas DataFrames,
                     each with at least one column.
        make_discrete (bool): If True, binarizes the input data before PCA.
    """
    matrix = pd.DataFrame({name: df.iloc[:, 0] for name, df in data.items()})

    if make_discrete:
        matrix = (matrix > 0.5).astype(int)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(matrix.T)

    pca_df = pd.DataFrame(reduced_data, columns=['PCA Component 1', 'PCA Component 2'])
    pca_df['Label'] = list(data.keys())

    def categorize_type(label):
        label_lower = label.lower()
        if label_lower.startswith('labeler_'):
            return 'Labeler'
        elif label_lower.startswith('model_'):
            return 'Model'
        # Add specific checks for methods based on your known names
        elif label_lower in ['loo_mean', 'chimera']: # Add any other specific method names
            return 'Mean'
        else:
            return 'Other' # Fallback for anything not categorized

    pca_df['Type'] = pca_df['Label'].apply(categorize_type)

    fig = px.scatter(pca_df,
                     x='PCA Component 1',
                     y='PCA Component 2',
                     text='Label',
                     color='Type', # Color points based on the new 'Type' column
                     symbol='Type', # Use different symbols for different types
                     title='2D Visualization of Labeler/Model/Method Similarity', # Updated title
                     labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2'},
                     hover_name='Label',
                     size_max=10,
                     color_discrete_map={
                         'Labeler': 'blue',
                         'Model': 'red',
                         'Method': 'green',
                         'Other': 'purple'
                     },
                     symbol_map={
                         'Labeler': 'circle',
                         'Model': 'square',
                         'Method': 'diamond',
                         'Other': 'x'
                     }
                    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Y=0", annotation_position="bottom right")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="X=0", annotation_position="top left")

    fig.update_traces(
        textposition='top center',
        marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey'))
    )

    fig.update_layout(
        showlegend=True,
        height=600,
        width=800,
        legend_title_text='Category' # Add a title for the legend
    )
    fig.show()


def plot_performance_on_video(folder_path: Path, models: Dict, labelers: Dict, fps: int = 25, bodyparts = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'], targets = ['obj_1', 'obj_2'], plot_tgt = "obj_1"):
    """
    Plots the performance of multiple models and labelers over time.

    Parameters:
        folder_path (Path): Path to the directory containing example video data.
        models (dict): Dictionary of model names and functions/lambdas to generate labels. 
                       Example: {"Simple": (model_simple, {}), "Wide": (model_wide, {"reshaping": True})}
        labelers (dict): Dictionary of labeler names and paths to their CSV files.
                         Example: {"lblr_A": "Example_Marian.csv"}
        fps (int): Frame rate of the video to calculate time in seconds. Default is 25.
        bodyparts (list): List of bodyparts used in the position data. Default is ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'].
        targets (list): List of target object names. Default is ['obj_1', 'obj_2'].
        plot_tgt (str): Name of the object column to plot. Default is 'obj_1'.
    """
    # Prepare dataset for the video
    X_view = pd.read_csv(folder_path / 'Example_position.csv').filter(regex='^(?!.*tail)')
    
    # Generate labels using models
    model_outputs = {}
    for key, path in models.items():
        print(f"Loading model from: {path}")
        model = tf.keras.models.load_model(path)

        # Determine if reshaping is needed
        reshaping = len(model.input_shape) == 3  # True if input is 3D

        if reshaping:
            past = future = model.input_shape[1] // 2
            output = use_model(X_view, model, targets, bodyparts, recentering = True, reshaping = True, past=past, future=future)
        
        else:
            output = use_model(X_view, model, targets, bodyparts, recentering=True)

        model_outputs[f"{key}"] = output

    # Load labeler data
    labeler_outputs = {}
    for labeler_name, labeler_file in labelers.items():
        labeler_outputs[labeler_name] = pd.read_csv(folder_path / labeler_file)

    # Create time axis
    time = np.arange(len(model_outputs[list(models.keys())[0]][plot_tgt])) / fps

    # Create a figure
    fig = go.Figure()

    # Add traces for labelers
    for idx, (labeler_name, labeler_data) in enumerate(labeler_outputs.items()):
        offset = 1 + 0.025 * (idx + 1)  # Incremental offset for visualization
        fig.add_trace(
            go.Scatter(
                x=time,
                y=[x * offset for x in labeler_data[plot_tgt]],
                mode='markers',
                name=labeler_name,
                marker=dict(color=f"hsl({idx * 60}, 70%, 50%)")
            )
        )

    # Add traces for models
    for model_name, model_output in model_outputs.items():
        fig.add_trace(
            go.Scatter(
                x=time,
                y=model_output[plot_tgt],
                mode='lines',
                name=model_name,
                line=dict(width=2)
            )
        )

    # Add horizontal line
    # fig.add_hline(y=0.5, line_dash="dash", line_color="black")

    # Update layout
    fig.update_layout(
        title=dict(text="Performance of the models & labelers"),
        xaxis_title="Time (s)",
        yaxis_title="Model output",
        showlegend=True,
    )

    # Show the plot
    fig.show()

def polar_graph(
    params_path: Path,
    positions: pd.DataFrame,
    label_1: pd.DataFrame,
    label_2: pd.DataFrame,
    target_name: str
) -> None:
    """
    Plots a polar graph with the distance and angle of approach to a single target object.
    
    Args:
        params_path (Path): Path to the YAML parameters file, containing
                            'geometric_analysis.roi_data.scale',
                            'geometric_analysis.distance', and
                            'geometric_analysis.orientation.degree'.
        positions (pd.DataFrame): DataFrame containing the positions of the bodyparts.
        label_1 (pd.DataFrame): DataFrame containing labels for the first comparison method.
        label_2 (pd.DataFrame): DataFrame containing labels for the second comparison method.
        target_name (str): The name of the target object to plot (e.g., "obj_1").
    """
    logger.info(f"Generating polar graph for object: {target_name}.")

    # Validate input DataFrames have necessary columns
    required_cols_positions = [f'nose_x', f'nose_y', f'head_x', f'head_y',
                               f'{target_name}_x', f'{target_name}_y']
    for col in required_cols_positions:
        if col not in positions.columns:
            logger.error(f"Missing required column in 'positions' DataFrame: {col}")
            return

    for lbl_df, lbl_name in zip([label_1, label_2], ['label_1', 'label_2']):
        if f"{target_name}" not in lbl_df.columns:
            logger.error(f"Missing required target column ({target_name}) in '{lbl_name}' DataFrame.")
            return

    params = load_yaml(params_path)
    scale = params.get("geometric_analysis", {}).get("roi_data", {}).get("scale", 1)
    distance = params.get("geometric_analysis", {}).get("distance", 2.5)
    degree = params.get("geometric_analysis", {}).get("orientation", {}).get("degree", 45)

    if not isinstance(scale, (int, float)) or scale <= 0:
        logger.warning(f"Invalid 'scale' value: {scale}. Using default scale of 1.0.")
        scale = 1.0
    if not isinstance(distance, (int, float)) or distance <= 0:
        logger.warning(f"Invalid 'distance' value: {distance}. Using default distance of 2.5.")
        distance = 2.5
    if not isinstance(degree, (int, float)):
        logger.warning(f"Invalid 'degree' value: {degree}. Using default degree of 45.")
        degree = 45

    # Create a copy to avoid modifying the original positions DataFrame
    positions_scaled = positions.copy()
    coords_cols = positions_scaled.filter(regex='_x|_y').columns
    positions_scaled[coords_cols] = positions_scaled[coords_cols] * (1.0 / scale)
    logger.info(f"Scaled positions by 1/{scale}.")
    
    # Extract positions of target object and bodyparts using Point class
    target_point = Point(positions_scaled, target_name)
    nose_point = Point(positions_scaled, 'nose')
    head_point = Point(positions_scaled, 'head')
    
    # Find distance from the nose to the target object
    dist = Point.dist(nose_point, target_point)
    
    head_nose_vec = Vector(head_point, nose_point, normalize=True)
    head_target_vec = Vector(head_point, target_point, normalize=True)
    
    # Find the angle between the head-nose and head-target vectors
    angle = Vector.angle(head_nose_vec, head_target_vec)
    logger.info("Calculated distances and angles.")

    """
    Plot
    """
    
    plt.rcParams['figure.figsize'] = [6, 6]  # Set the figure size
    plt.rcParams['font.size'] = 12
    
    # Set start and finish frames
    a, b = 0, -1
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    
    # Set title for the first subplot
    ax.set_title(f"Approach to {target_name}")
    
    # Determine colors and alpha based on labels
    colors_1 = ['#E53935' if label >= 0.5 else 'gray' for label in label_1[f"{target_name}"][a:b]]
    alpha_1 = [0.8 if label >= 0.5 else 0.3 for label in label_1[f"{target_name}"][a:b]]
    
    colors_2 = ['#3F51B5' if label >= 0.5 else 'gray' for label in label_2[f"{target_name}"][a:b]]
    alpha_2 = [0.8 if label >= 0.5 else 0.3 for label in label_2[f"{target_name}"][a:b]]
    
    # Plot data points
    ax.scatter((angle[a:b] + 90) / 180 * np.pi, dist[a:b], c=colors_1, s=6, alpha=alpha_1)
    ax.scatter(-(angle[a:b] - 90) / 180 * np.pi, dist[a:b], c=colors_2, s=6, alpha=alpha_2)
    
    angle_start = (np.pi/2) - np.deg2rad(degree)  # Convert degrees to radians
    angle_end = np.pi/2  # 90° in radians

    ang_plot = np.linspace(angle_start, angle_end, 25).tolist()
    
    ax.plot([0] + ang_plot + [0], [0] + [distance] * 25 + [0], c="k", linestyle='dashed', linewidth=4)
    
    ax.set_ylim([0, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1 cm", "2 cm", "3 cm", "4 cm"])
    ax.set_xticks(
        [0, 45 / 180 * np.pi, 90 / 180 * np.pi, 135 / 180 * np.pi, np.pi, 225 / 180 * np.pi, 270 / 180 * np.pi,
         315 / 180 * np.pi])
    ax.set_xticklabels(["  90°", "45°", "0°", "45°", "90°  ", "135°    ", "180°", "    135°"])
    
    # Create legend handles for both subplots
    legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Automatic'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Manual'),
    Line2D([0], [0], color='k', linestyle='dashed', linewidth=2, label='Geometric')]

    # Add legend to the figure
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()