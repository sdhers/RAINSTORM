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
    GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler,
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from pathlib import Path
import datetime

from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Model Building Functions

def build_RNN(modeling_path: Path, model_dict: Dict[str, np.ndarray]) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM (RNN) model for binary classification.

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
        raise KeyError("model_dict must include 'X_tr_wide' to infer input shape.")
    x_sample = model_dict['X_tr_wide']
    if x_sample.ndim != 3:
        raise ValueError("'X_tr_wide' must be 3D (batch, time, features) to define input shape.")

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

    # Apply GlobalMaxPooling1D to convert the sequence of features
    # from the last LSTM layer into a single fixed-size vector.
    # This layer takes the maximum value across the time dimension for each feature.
    pooled_output = GlobalMaxPooling1D(name="global_max_pooling")(x)

    # Output layer for binary classification
    # Uses a sigmoid activation to output a probability between 0 and 1.
    binary_out = Dense(1, activation='sigmoid', name="binary_out")(pooled_output)

    # Create the Keras Model
    model = Model(inputs, binary_out, name="CleanedBidirectionalRNN")

    # Compile the model
    # Adam optimizer is a good default for many tasks.
    # 'binary_crossentropy' is the standard loss for binary classification.
    # 'accuracy' is a common metric to monitor performance.
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr) # Initial LR for optimizer, will be overridden by scheduler
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Print a summary of the model architecture
    model.summary()
    return model

# %% Model Training Functions

def train_RNN(modeling_path: Path, model: tf.keras.Model, model_dict: Dict[str, np.ndarray], model_name: str) -> Any:
    """
    Trains the given RNN model using the provided data splits.
    It incorporates a custom sigmoid-shaped learning rate schedule,
    early stopping, model checkpointing, TensorBoard logging, and
    ReduceLROnPlateau for further learning rate adjustments after warmup.

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
    warmup_epochs = rnn_conf.get("warmup_epochs", 10) # Number of epochs for learning rate warmup/cooldown phase
    initial_lr = rnn_conf.get("initial_lr", 1e-5) # Starting and final learning rate during warmup phase
    peak_lr = rnn_conf.get("peak_lr", 1e-4) # Maximum learning rate during warmup phase
    batch_size = rnn_conf.get("batch_size", 64) # Number of samples per gradient update
    patience = rnn_conf.get("patience", 10) # Number of epochs with no improvement after which training will be stopped

    # Define the save folder for logs and checkpoints
    save_folder = Path(load_yaml(modeling_path).get("path"))

    logger.info(f"🚀 Starting training for model: {model_name}")

    # Learning Rate Schedule: Sigmoid-shaped warmup to peak_lr, then back to initial_lr
    # within warmup_epochs. After warmup_epochs, learning rate remains at initial_lr.
    def lr_schedule(epoch: int, lr: float) -> float:
        """
        Custom learning rate scheduler function with a sigmoid-shaped increase and
        decrease within the warmup period, followed by a constant initial_lr.
        """
        sigmoid_sharpness = 8 # Controls the steepness of the sigmoid curve

        if epoch < warmup_epochs:
            if warmup_epochs == 0: # Handle case where warmup_epochs is 0
                return initial_lr

            if epoch <= warmup_epochs / 2:
                # Rising sigmoid curve from initial_lr to peak_lr
                # Normalize epoch to [0, 1] for sigmoid input
                progress = epoch / (warmup_epochs / 2)
                # Apply sigmoid function. Shifted by -0.5 to center the steep part at 0.5.
                sigmoid_val = 1 / (1 + np.exp(-sigmoid_sharpness * (progress - 0.5)))
                return initial_lr + (peak_lr - initial_lr) * sigmoid_val
            else:
                # Falling sigmoid curve from peak_lr back to initial_lr
                # Normalize epoch from the midpoint to the end of warmup to [0, 1]
                progress = (epoch - (warmup_epochs / 2)) / (warmup_epochs / 2)
                # Apply sigmoid function and invert for decreasing curve
                sigmoid_val = 1 / (1 + np.exp(-sigmoid_sharpness * (progress - 0.5)))
                return initial_lr + (peak_lr - initial_lr) * (1 - sigmoid_val)
        else:
            # After the warmup phase, allow ReduceLROnPlateau to manage the LR.
            # If ReduceLROnPlateau has already changed the LR, respect that change.
            # Otherwise, keep it at initial_lr.
            # We assume 'lr' argument passed to lr_schedule is the LR from the previous epoch.
            # If ReduceLROnPlateau reduced it, 'lr' will be the reduced value.
            return lr if lr < initial_lr else initial_lr

    # Callbacks for training process control and monitoring
    
    # 1. Learning Rate Scheduler: Adjusts the learning rate based on the custom schedule.
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1) # Set verbose to 1 to log LR

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

    # 5. ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.
    #    It will primarily act after the initial LR schedule stabilizes at initial_lr.
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='val_loss', # Metric to monitor
        factor=0.5, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=patience // 2, # Number of epochs with no improvement after which learning rate will be reduced
        verbose=1, # Prints messages when triggered
        min_lr=1e-7 # Lower bound on the learning rate
    )

    # List of all callbacks to be used during training
    callbacks = [
        lr_scheduler,
        early_stopping,
        tensorboard_callback,
        model_checkpoint,
        reduce_lr_on_plateau
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

