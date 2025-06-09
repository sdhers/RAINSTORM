# rainstorm/__init__.py

# Import core functions/classes from submodules to make them accessible
# directly from the 'rainstorm.modeling' package.
from .data_preparation import create_colabels, prepare_data, focus
from .geometry_utils import Point, Vector
from .model_management import create_modeling, load_split, save_split, build_and_run_models, load_modeling_config, save_model
from .model_building import build_RNN
from .model_training import split_tr_ts_val, train_RNN
from .model_evaluating import evaluate, build_evaluation_dict, create_chimera_and_loo_mean, calculate_cosine_sim, plot_PCA, plot_history, plot_performance_on_video

# You might want to add a __version__ here for package versioning
__version__ = "0.1.0"

# Optional: Configure logging for the rainstorm package
import logging
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if this is imported multiple times
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')