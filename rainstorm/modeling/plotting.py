import logging
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
from pathlib import Path

from .model_building import use_model
from .utils import configure_logging
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