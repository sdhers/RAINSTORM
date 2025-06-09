# rainstorm/model_training.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler,
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from typing import Dict, Any, List
from pathlib import Path
import logging
import datetime

# Import necessary utilities from the main utils file
from .utils import reshape, load_yaml

logger = logging.getLogger(__name__)

# %% Model Training Functions

def split_tr_ts_val(modeling_path: Path, df: pd.DataFrame, bodyparts: List[str]) -> Dict[str, np.ndarray]:
    """
    Splits the data into training, validation, and test sets.

    Args:
        modeling_path (Path): Path to modeling.yaml with split configuration.
        df (pd.DataFrame): Input DataFrame containing position and label data.
        bodyparts (List[str]): List of body parts to include as features.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the split data arrays:
                               'X_tr_wide', 'y_tr', 'X_val_wide', 'y_val', 'X_ts_wide', 'y_ts'.
    """
    modeling = load_yaml(modeling_path)
    split_conf = modeling.get("split", {})
    test_split_ratio = split_conf.get("test", 0.15)
    validation_split_ratio = split_conf.get("validation", 0.15)
    
    rnn_conf = modeling.get("RNN", {})
    width_conf = rnn_conf.get("width", {})
    past = width_conf.get("past", 0)
    future = width_conf.get("future", 0)
    broad = width_conf.get("broad", 1)

    logger.info("📊 Splitting data into training, validation, and test sets...")

    # Select features (body parts) and labels
    features_cols = [f"{bp}_x" for bp in bodyparts] + [f"{bp}_y" for bp in bodyparts]
    X = df[features_cols]
    y = df["labels"].values

    # Determine split indices
    num_samples = len(X)
    num_test = int(num_samples * test_split_ratio)
    num_val = int(num_samples * validation_split_ratio)
    num_train = num_samples - num_test - num_val

    # Ensure reproducibility with a fixed seed if needed for debugging
    # np.random.seed(42)

    # Randomly select indices for test and validation, then use remaining for training
    indices = np.random.permutation(num_samples)
    
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    # Apply splits to original X and y
    X_ts, y_ts = X.iloc[test_indices], y[test_indices]
    X_val, y_val = X.iloc[val_indices], y[val_indices]
    X_tr, y_tr = X.iloc[train_indices], y[train_indices]

    logger.info(f"  Training samples: {len(X_tr)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Test samples: {len(X_ts)}")

    # Reshape for RNN
    X_tr_wide = reshape(X_tr, past, future, broad)
    X_val_wide = reshape(X_val, past, future, broad)
    X_ts_wide = reshape(X_ts, past, future, broad)
    
    logger.info(f"  Reshaped X_tr_wide shape: {X_tr_wide.shape}")
    logger.info(f"  Reshaped X_val_wide shape: {X_val_wide.shape}")
    logger.info(f"  Reshaped X_ts_wide shape: {X_ts_wide.shape}")

    return {
        'X_tr_wide': X_tr_wide, 'y_tr': y_tr,
        'X_val_wide': X_val_wide, 'y_val': y_val,
        'X_ts_wide': X_ts_wide, 'y_ts': y_ts
    }


def train_RNN(modeling_path: Path, model: tf.keras.Model, model_dict: Dict[str, np.ndarray], model_name: str) -> Any:
    """
    Trains the given RNN model using the provided data splits and saves the training history.

    Args:
        modeling_path (Path): Path to modeling.yaml with training configuration.
        model (tf.keras.Model): The compiled Keras RNN model to train.
        model_dict (Dict[str, np.ndarray]): Dictionary containing split data arrays
                                            ('X_tr_wide', 'y_tr', 'X_val_wide', 'y_val').
        model_name (str): Name for the model, used for TensorBoard logs and checkpoints.

    Returns:
        tf.keras.callbacks.History: The training history object.
    """
    rnn_conf = load_yaml(modeling_path).get("RNN", {})
    total_epochs = rnn_conf.get("total_epochs", 100)
    warmup_epochs = rnn_conf.get("warmup_epochs", 10)
    initial_lr = rnn_conf.get("initial_lr", 1e-5)
    peak_lr = rnn_conf.get("peak_lr", 1e-4)
    batch_size = rnn_conf.get("batch_size", 64)
    patience = rnn_conf.get("patience", 10)
    
    save_folder = Path(load_yaml(modeling_path).get("path"))

    logger.info(f"🚀 Starting training for model: {model_name}")

    # Learning Rate Schedule (linear warmup, then decay)
    def lr_schedule(epoch, lr):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Cosine decay or a simple exponential decay after warmup
            # For simplicity, using a fixed decay rate after warmup to avoid
            # dependency on total_epochs for final LR.
            decay_epochs = total_epochs - warmup_epochs
            # Adjust `drop` and `epochs_drop` as needed for desired decay curve
            drop = 0.5
            epochs_drop = 10.0
            return initial_lr * (drop ** (epoch / epochs_drop))
            # A more sophisticated approach might be:
            # cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / decay_epochs))
            # return peak_lr * cosine_decay

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    log_dir = save_folder / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = save_folder / "checkpoints" / f"{model_name}_best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    # ReduceLROnPlateau can be an alternative or complement to LearningRateScheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)

    callbacks = [
        lr_scheduler,
        early_stopping,
        tensorboard_callback,
        model_checkpoint,
        reduce_lr # Added for robustness
    ]

    history = model.fit(
        model_dict['X_tr_wide'], model_dict['y_tr'],
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(model_dict['X_val_wide'], model_dict['y_val']),
        callbacks=callbacks,
        verbose=1
    )

    logger.info(f"✅ Training for model '{model_name}' completed.")
    return history

