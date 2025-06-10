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
    Activation,
    Multiply,
    Lambda
)
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler,
    ModelCheckpoint, TensorBoard
)
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from pathlib import Path
import datetime

from .utils import load_yaml, configure_logging, reshape, recenter
configure_logging()

logger = logging.getLogger(__name__)

# %% Model Building Functions

def attention_pooling_block(x: tf.Tensor, name_prefix: str = "attn_pool") -> tf.Tensor:
    """
    Functional attention pooling: softmax over time, weighted sum.
    This block takes a sequence (batch, timesteps, features) and returns
    a single vector per batch element (batch, features) by
    applying an attention mechanism to weigh the importance of each timestep.

    Args:
        x: Input tensor with shape (batch_size, timesteps, features).
        name_prefix: Prefix for naming the Keras layers within this block.
    Returns:
        tf.Tensor: Output tensor with shape (batch_size, features).
    """
    # Compute scores for each timestep (how important is this timestep?)
    score = Dense(1, name=f"{name_prefix}_score")(x)
    
    # Apply softmax to get attention weights that sum to 1 across timesteps
    weights = Activation('softmax', name=f"{name_prefix}_weights")(score)
    
    # Multiply the original features by the attention weights
    weighted = Multiply(name=f"{name_prefix}_weighted")([x, weights])
    
    # Sum along the timesteps dimension to get a single context vector
    return Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f"{name_prefix}_sum")(weighted)

def build_RNN(modeling_path: Path, model_dict: Dict[str, np.ndarray]) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM (RNN) model for binary classification.
    The model processes sequences and uses an attention pooling mechanism
    to derive a single vector representation for classification.

    Args:
        modeling_path (Path): Path to a YAML file containing modeling parameters,
                              specifically the 'RNN' configuration.
        model_dict (Dict[str, np.ndarray]): Dictionary containing 'X_tr_wide'
                                            which is a sample of the training data
                                            shaped (batch, time, features) to infer
                                            input dimensions.

    Returns:
        tf.keras.Model: Compiled Keras RNN model ready for training.
    """
    modeling_conf = load_yaml(modeling_path)
    rnn_conf = modeling_conf.get("RNN", {})

    # Model configuration parameters
    units = rnn_conf.get("units", [16, 24, 32, 24, 16, 8]) # Number of units in each LSTM layer
    dropout_rate = rnn_conf.get("dropout", 0.2) # Dropout rate for regularization
    initial_lr = rnn_conf.get("initial_lr", 1e-5) # Initial learning rate for the optimizer

    # Validate input data shape
    if 'X_tr_wide' not in model_dict:
        logger.error("model_dict must include 'X_tr_wide' to infer input shape.")
        return None
    x_sample = model_dict['X_tr_wide']
    if x_sample.ndim != 3:
        logger.error("'X_tr_wide' must be 3D (batch, time, features) to define input shape.")
        return None

    timesteps, features = x_sample.shape[1], x_sample.shape[2]
    
    # Define the input layer of the model
    inputs = Input(shape=(timesteps, features), name="input_sequence")
    x = inputs

    # Build stacked Bidirectional LSTM layers
    # Each LSTM layer returns sequences, allowing the next layer to process them.
    for i, u in enumerate(units):
        # Bidirectional LSTM to capture dependencies in both forward and backward directions
        bilstm = Bidirectional(LSTM(u, return_sequences=True, name=f"bilstm_{i}"))(x)
        
        # Batch Normalization to stabilize activations and speed up training
        bn = BatchNormalization(name=f"bn_{i}")(bilstm)
        
        # Dropout for regularization to prevent overfitting
        drop = Dropout(dropout_rate, name=f"dropout_{i}")(bn)
        
        x = drop # Output of current layer becomes input for the next

    # Apply global attention pooling to convert the sequence of features from the last LSTM layer into a single fixed-size vector.
    attention_pooled = attention_pooling_block(x, name_prefix="final_attention_pooling")

    # Output layer for binary classification
    # Uses a sigmoid activation to output a probability between 0 and 1.
    binary_out = Dense(1, activation='sigmoid', name="binary_out")(attention_pooled)

    # Create the Keras Model
    model = Model(inputs, binary_out, name="CleanedBidirectionalRNN")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr) # Initial LR for optimizer, will be overridden by scheduler
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info("✅ RNN model built successfully.")
    model.summary()
    return model

# %% Model Training Functions

def train_RNN(modeling_path: Path, model: tf.keras.Model, model_dict: Dict[str, np.ndarray], model_name: str) -> Any:
    """
    Trains the given RNN model using the provided data splits.
    It incorporates a learning rate schedule, early stopping,
    model checkpointing, and TensorBoard logging.

    Args:
        modeling_path (Path): Path to a YAML file containing training configuration
                              parameters under the 'RNN' key.
        model (tf.keras.Model): The compiled Keras RNN model to train.
        model_dict (Dict[str, np.ndarray]): Dictionary containing the data splits:
                                            'X_tr_wide', 'y_tr' (training data),
                                            'X_val_wide', 'y_val' (validation data).
        model_name (str): A name for the model, used for organizing TensorBoard logs
                          and saved model checkpoints.

    Returns:
        tf.keras.callbacks.History: The training history object, containing
                                    loss and metric values per epoch.
    """
    rnn_conf = load_yaml(modeling_path).get("RNN", {})

    # Training configuration parameters
    total_epochs = rnn_conf.get("total_epochs", 100) # Maximum number of training epochs
    warmup_epochs = rnn_conf.get("warmup_epochs", 10) # Number of epochs for learning rate warmup
    initial_lr = rnn_conf.get("initial_lr", 1e-5) # Starting learning rate for warmup
    peak_lr = rnn_conf.get("peak_lr", 1e-4) # Maximum learning rate during warmup
    batch_size = rnn_conf.get("batch_size", 64) # Number of samples per gradient update
    patience = rnn_conf.get("patience", 10) # Number of epochs with no improvement after which training will be stopped

    # Define the save folder for logs and checkpoints
    save_folder = Path(load_yaml(modeling_path).get("path"))

    logger.info(f"🚀 Starting training for model: {model_name}")

    # Learning Rate Schedule: Linear Warmup followed by Cosine Decay
    def lr_schedule(epoch: int, lr: float) -> float:
        """
        Custom learning rate scheduler function.
        Linearly increases LR during warmup_epochs, then decays with a cosine annealing.
        """
        if epoch < warmup_epochs:
            # Linear warmup phase
            return initial_lr + (peak_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Cosine decay phase
            decay_epochs = warmup_epochs # decay and warmup share the same duration (for simplicity)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / decay_epochs))
            return peak_lr * cosine_decay

    # Callbacks for training process control and monitoring
    
    # 1. Learning Rate Scheduler: Adjusts the learning rate based on the custom schedule.
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

    # 2. Early Stopping: Stops training if validation loss does not improve for 'patience' epochs.
    #    Restores the weights from the epoch with the best validation loss.
    early_stopping = EarlyStopping(
        monitor='val_loss', # Metric to monitor
        patience=patience, # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Restores model weights from the epoch with the best value of the monitored quantity
        verbose=1 # Prints messages when triggered
    )

    # 3. TensorBoard Callback: Logs metrics and graph information for visualization in TensorBoard.
    log_dir = save_folder / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S") / model_name
    log_dir.mkdir(parents=True, exist_ok=True) # Create log directory if it doesn't exist
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 4. Model Checkpoint: Saves the model's weights (or full model) only when
    #    'val_loss' improves, ensuring only the best performing model is kept.
    checkpoint_path = save_folder / "checkpoints" / f"{model_name}_best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True) # Create checkpoint directory if it doesn't exist
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, # Path to save the model
        monitor='val_loss', # Metric to monitor for saving
        save_best_only=True, # Only save when the monitored quantity improves
        verbose=0 # Suppress verbose output during saving
    )

    # List of all callbacks to be used during training
    callbacks = [
        lr_scheduler,
        early_stopping,
        tensorboard_callback,
        model_checkpoint
    ]

    # Start training the model
    history = model.fit(
        model_dict['X_tr_wide'], model_dict['y_tr'], # Training data and labels
        epochs=total_epochs, # Maximum number of epochs to train
        batch_size=batch_size, # Number of samples per batch
        validation_data=(model_dict['X_val_wide'], model_dict['y_val']), # Validation data
        callbacks=callbacks, # List of callbacks to apply during training
        verbose=1 # Show progress bar and epoch details
    )

    logger.info(f"✅ Training for model '{model_name}' completed.")
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