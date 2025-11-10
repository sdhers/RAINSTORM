"""
RAINSTORM - Modeling

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.modeling' package.
"""

# Import and configure logging first
from ..utils import configure_logging
configure_logging(module_name='modeling')

# Import the params_editor module from the submodule
from ..prepare_positions import open_params_editor

from .create_colabels import create_colabels
from .data_handling import prepare_data, focus, split_tr_ts_val, save_split, load_split
from .plotting import plot_example_data, plot_history, plot_lr_schedule, plot_cosine_sim, plot_PCA, plot_performance_on_video, polar_graph
from .model_building import build_RNN, train_RNN, save_model
from .model_evaluating import evaluate, build_evaluation_dict, create_chimera_and_loo_mean, build_model_paths, build_and_run_models
from .automatic_analysis import create_autolabels, prepare_label_comparison, accuracy_scores
from .colabels_gui import open_colabels_gui

import tensorflow as tf

# Print GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"rainstorm.modeling successfully imported. GPU devices detected: {gpu_devices}")

# Define __all__ for explicit export
__all__ = [
    'configure_logging',
    'open_params_editor',
    'create_colabels',
    'prepare_data',
    'focus',
    'split_tr_ts_val',
    'save_split',
    'load_split',
    'plot_example_data',
    'plot_history',
    'plot_lr_schedule',
    'plot_cosine_sim',
    'plot_PCA',
    'plot_performance_on_video',
    'polar_graph',
    'build_RNN',
    'train_RNN',
    'save_model',
    'evaluate',
    'build_evaluation_dict',
    'create_chimera_and_loo_mean',
    'build_model_paths',
    'build_and_run_models',
    'create_autolabels',
    'prepare_label_comparison',
    'accuracy_scores',
    'open_colabels_gui',
]