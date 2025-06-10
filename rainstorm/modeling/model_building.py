# rainstorm/model_building.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Bidirectional,
    LSTM,
    BatchNormalization,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Cropping1D,
    Activation,
    Multiply,
    Lambda
)
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler,
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
from pathlib import Path
import datetime

from .utils import load_yaml, configure_logging, reshape, recenter
configure_logging()

logger = logging.getLogger(__name__)

# %% Model Building Functions

def generate_slice_plan(timesteps: int, num_layers: int) -> List[int]:
    """
    Generate a per-layer cropping plan to reduce timesteps down to 1.

    Args:
        timesteps: Initial sequence length.
        num_layers: Number of RNN layers.
    Returns:
        List of ints: timesteps to remove at each layer.
    """
    total_to_remove = max(0, timesteps - 2)
    plan = [0] * num_layers
    for i in range(total_to_remove):
        plan[i % num_layers] += 1
    return plan

def attention_pooling_block(x: tf.Tensor, name_prefix: str = "attn_pool") -> tf.Tensor:
    """
    Functional attention pooling: softmax over time, weighted sum.

    Args:
        x: (batch, timesteps, features)
        name_prefix: prefix for naming layers
    Returns:
        (batch, features)
    """
    score = Dense(1, name=f"{name_prefix}_score")(x)
    weights = Activation('softmax', name=f"{name_prefix}_weights")(score)
    weighted = Multiply(name=f"{name_prefix}_weighted")([x, weights])
    return Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f"{name_prefix}_sum")(weighted)

def build_RNN(modeling: Path, model_dict: Dict[str, np.ndarray]) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM (RNN) model with optional attention mechanism.

    Args:
        modeling (Path): Path to dictionary containing modeling parameters, especially 'RNN' configuration.
        model_dict (Dict[str, np.ndarray]): Contains 'X_tr_wide' shaped (batch, time, features).

    Returns:
        tf.keras.Model: Compiled Keras RNN model.
    """
    modeling = load_yaml(modeling)
    rnn_conf = modeling.get("RNN", {})
    units = rnn_conf.get("units", [16, 24, 32, 24, 16, 8])
    dropout_rate = rnn_conf.get("dropout", 0.2)
    initial_lr = rnn_conf.get("initial_lr", 1e-5)

    # Validate input
    if 'X_tr_wide' not in model_dict:
        raise KeyError("model_dict must include 'X_tr_wide'.")
    x_sample = model_dict['X_tr_wide']
    if x_sample.ndim != 3:
        raise ValueError("'X_tr_wide' must be 3D (batch, time, features).")

    timesteps, features = x_sample.shape[1], x_sample.shape[2]
    inputs = Input(shape=(timesteps, features), name="input_sequence")
    x = inputs

    # Plan cropping to reach 1 timestep
    slice_plan = generate_slice_plan(timesteps, len(units))
    current_steps = timesteps

    # RNN layers with Batch Normalization and Dropout
    for i, u in enumerate(units):
        bilstm = Bidirectional(LSTM(u, return_sequences=True), name=f"bilstm_{i}")(x)
        bn = BatchNormalization(name=f"bn_{i}")(bilstm)
        drop = Dropout(dropout_rate, name=f"dropout_{i}")(bn)
        
        # Add a self-attention mechanism after the LSTM layer
        # Layer Normalization before attention
        norm_attn = LayerNormalization(name=f"attn_norm_{i}")(drop)
        
        # Multi-Head Attention
        # num_heads should be a factor of units, default to 1 if units is small or not divisible
        num_heads = max(1, u // 8) # A common heuristic for attention heads
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=u, name=f"attn_{i}")(
            norm_attn, norm_attn
        )
        
        # Add and Normalize (Skip connection for attention)
        add_attn = Add(name=f"attn_add_{i}")([drop, attention_output])

        # Controlled cropping
        remove = slice_plan[i]
        if remove > 0 and current_steps > 1:
            left = remove // 2
            right = remove - left
            x = Cropping1D(cropping=(left, right), name=f"crop_{i}")(add_attn)
            current_steps -= remove

    # Global attention pooling to convert sequence to a single vector
    attention_pooled = attention_pooling_block(x, name_prefix="attention_pooling")

    # Output layer
    binary_out = Dense(1, activation='sigmoid', name="binary_out")(attention_pooled)

    model = Model(inputs, binary_out, name="ModularBidirectionalRNN")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr) # Learning rate is set by LearningRateScheduler, so this is just a default.
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

# %% Model Training Functions

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
    print(f"🚀 Starting training for model: {model_name}")

    # Learning Rate Schedule (linear warmup, then decay)
    def lr_schedule(epoch, lr):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            decay_epochs = total_epochs - warmup_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / decay_epochs))
            return peak_lr * cosine_decay

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
        reduce_lr
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
    print(f"✅ Training for model '{model_name}' completed.")
    return history

# %% Model Management Functions

def save_model(modeling_path: Path, model: tf.keras.Model, model_name: str) -> None:
    """
    Save a trained TensorFlow model.

    Args:
        modeling_path (Path): Path to modeling.yaml to get the save folder.
        model (tf.keras.Model): The trained Keras model to save.
        model_name (str): Name for the saved model file.
    """
    modeling = load_yaml(modeling_path)
    save_folder = Path(modeling.get("path")) / 'trained_models'
    save_folder.mkdir(parents=True, exist_ok=True)
    
    filepath = save_folder / f"{model_name}.keras"
    model.save(filepath)
    logger.info(f"✅ Model '{model_name}' saved to: {filepath}")
    print(f"Model '{model_name}' saved to: {filepath}")

def use_model(position, model, objects = ['tgt'], bodyparts = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'], recentering = False, reshaping = False, past: int = 3, future: int = 3, broad: float = 1.7):
    
    if recentering:
        position = pd.concat([recenter(position, obj, bodyparts) for obj in objects], ignore_index=True)

    if reshaping:
        position = np.array(reshape(position, past, future, broad))
    
    pred = model.predict(position) # Use the model to predict the labels
    pred = pred.flatten()
    pred = pd.DataFrame(pred, columns=['predictions'])

    # Smooth the predictions
    pred.loc[pred['predictions'] < 0.1, 'predictions'] = 0  # Set values below 0.3 to 0
    #pred.loc[pred['predictions'] > 0.98, 'predictions'] = 1
    #pred = smooth_columns(pred, ['predictions'], gauss_std=0.2)

    n_objects = len(objects)

    # Calculate the length of each fragment
    fragment_length = len(pred) // n_objects

    # Create a list to hold each fragment
    fragments = [pred.iloc[i*fragment_length:(i+1)*fragment_length].reset_index(drop=True) for i in range(n_objects)]

    # Concatenate fragments along columns
    labels = pd.concat(fragments, axis=1)

    # Rename columns
    labels.columns = [f'{obj}' for obj in objects]
    
    return labels

def build_and_run_models(modeling_path: Path, model_paths: Dict[str, Path], position_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Loads specified models, prepares input data for each, and generates predictions.

    Args:
        modeling_path (Path): Path to the modeling.yaml file.
        model_paths (Dict[str, Path]): Dictionary mapping model names to their file paths.
        position_df (pd.DataFrame): DataFrame containing the full position data.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping model names (prefixed with 'model_')
                               to their prediction arrays.
    """
    modeling = load_yaml(modeling_path)
    bodyparts = modeling.get("bodyparts", [])
    target = modeling.get("colabels", {}).get("target", 'tgt')
    targets = [target] # Because 'use_model' only accepts a list of targets

    X_all = position_df.copy()
    models_dict = {}
    
    for key, path in model_paths.items():
        logger.info(f"Loading model from: {path}")
        print(f"Loading model from: {path}")
        model = tf.keras.models.load_model(path)

        # Determine if reshaping is needed
        reshaping = len(model.input_shape) == 3  # True if input is 3D

        if reshaping:
            past = future = model.input_shape[1] // 2
            output = use_model(X_all, model, targets, bodyparts, recentering=True, reshaping=True, past=past, future=future)
        
        else:
            output = use_model(X_all, model, targets, bodyparts, recentering=True)

        # Store the result in the dictionary
        models_dict[f"model_{key}"] = output

    return models_dict