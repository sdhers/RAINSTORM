"""
RAINSTORM - Modeling

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.modeling' package.
"""

# Import and configure logging first
from .utils import load_yaml, configure_logging
configure_logging()

from .geometric_classes import Point, Vector
from .create_colabels import create_colabels
from .create_modeling_file import create_modeling
from .data_handling import prepare_data, focus, split_tr_ts_val, save_split, load_split
from .plotting import plot_example_data, plot_history, plot_cosine_sim, plot_PCA, plot_performance_on_video
from .model_building import save_model, build_RNN, train_RNN, build_and_run_models
from .model_evaluating import evaluate, build_evaluation_dict, create_chimera_and_loo_mean

import tensorflow as tf
print(f"rainstorm.modeling successfully imported. GPU devices detected: {tf.config.list_physical_devices('GPU')}")

# Define __all__ for explicit export (optional but good practice)
__all__ = [
]