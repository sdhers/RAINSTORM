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
    GlobalMaxPooling1D,
    Conv1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler,
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path
import datetime

from ..utils import configure_logging, load_yaml
configure_logging()
logger = logging.getLogger(__name__)

# %% Model Building Functions

def build_RNN(params_path: Path, model_dict: Dict[str, np.ndarray]) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM (RNN) model for binary classification.

    Args:
        params_path (Path): Path to the YAML file containing modeling parameters,
                              specifically the 'ANN' configuration.
        model_dict (Dict[str, np.ndarray]): Dictionary containing 'X_tr_wide'
                                            which is a sample of the training data
                                            shaped (batch, time, features) to infer
                                            input dimensions.

    Returns:
        tf.keras.Model: Compiled Keras RNN model ready for training.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    ANN = modeling.get("ANN") or {}

    # Model configuration parameters
    units = ANN.get("units") or [32, 16, 8] # Number of units in each LSTM layer
    dropout_rate = float(ANN.get("dropout") or 0.2) # Dropout rate for regularization
    initial_lr = float(ANN.get("initial_lr") or 1e-5) # Initial learning rate for the optimizer

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

    # Use causal padding to maintain temporal order for short windows
    x = Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu', name="conv1d_motion")(x)
    x = BatchNormalization(name="bn_conv")(x)
    x = Dropout(dropout_rate, name="dropout_conv")(x)
    
    # Build stacked Bidirectional LSTM layers. Each LSTM layer returns sequences, allowing the next layer to process them.
    for i, u in enumerate(units):
        # Bidirectional LSTM to capture dependencies in both forward and backward directions
        bilstm = Bidirectional(LSTM(u, return_sequences=True, name=f"bilstm_{i}"))(x)
        
        # Batch Normalization to stabilize activations and speed up training
        bn = BatchNormalization(name=f"bn_{i}")(bilstm)
        
        # Dropout for regularization to prevent overfitting
        drop = Dropout(dropout_rate, name=f"dropout_{i}")(bn)
        
        x = drop # Output of current layer becomes input for the next

    # Apply GlobalMaxPooling1D to convert the sequence of features from the last LSTM layer into a single fixed-size vector.
    x = GlobalMaxPooling1D(name="global_max_pooling")(x)

    # Dense classification head
    x = Dense(8, activation='relu', name="dense_8")(x)
    x = Dropout(dropout_rate, name="dropout_dense_8")(x)

    x = Dense(4, activation='relu', name="dense_4")(x)
    x = Dropout(dropout_rate, name="dropout_dense_4")(x)

    # Output layer for binary classification, uses a sigmoid activation to output a probability between 0 and 1.
    binary_out = Dense(1, activation='sigmoid', name="binary_out")(x)

    # Create the Keras Model
    model = Model(inputs, binary_out, name="CleanedBidirectionalRNN")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Print a summary of the model architecture
    model.summary()
    return model

# %% Model Training Functions

def train_RNN(params_path: Path, model: tf.keras.Model, model_dict: Dict[str, np.ndarray], model_name: str) -> Any:
    """
    Trains a Conv1D + BiLSTM + dense head model for behavior classification.
    Uses linear warmup + cosine decay, gradient clipping, class weighting, and adaptive batch sizing.

    Args:
        params_path (Path): Path to the YAML file containing training configuration
                              parameters under the 'ANN' key.
        model (tf.keras.Model): The compiled Keras model to train.
        model_dict (Dict[str, np.ndarray]): Dictionary containing the data splits:
                                            'X_tr_wide', 'y_tr' (training data),
                                            'X_val_wide', 'y_val' (validation data).
        model_name (str): A name for the model, used for organizing TensorBoard logs
                          and saved model checkpoints.

    Returns:
        tf.keras.callbacks.History: The training history object.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    ANN = modeling.get("ANN") or {}

    # Training configuration parameters
    total_epochs = int(ANN.get("total_epochs")) or 100
    warmup_epochs = int(ANN.get("warmup_epochs")) or 8
    initial_lr = float(ANN.get("initial_lr")) or 1e-5
    peak_lr = float(ANN.get("peak_lr")) or 1e-4
    batch_size = int(ANN.get("batch_size")) or 32
    patience = int(ANN.get("patience")) or 10

    X_tr = model_dict['X_tr_wide']
    seq_length = X_tr.shape[1]

    # Define the save folder for logs and checkpoints
    save_folder = Path(modeling.get("models_path"))

    logger.info(f"🚀 Starting enhanced training for model: {model_name}")
    logger.info(f"📊 Dataset: {X_tr.shape[0]} samples, {seq_length} timesteps, {X_tr.shape[2]} features")

    # Learning Rate Schedule: Linear warmup + cosine decay
    class CustomLRScheduler:
        def __init__(self, initial_lr, peak_lr, warmup_epochs):
            self.initial_lr = float(initial_lr)
            self.peak_lr = float(peak_lr)
            self.warmup_epochs = int(warmup_epochs)
        
        def __call__(self, epoch: int, lr: float) -> float:
            if epoch < self.warmup_epochs:
                if self.warmup_epochs == 0: # Handle case where warmup_epochs is 0
                    return self.initial_lr
                # Warmup
                return self.initial_lr + (self.peak_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            elif epoch < self.warmup_epochs*2:
                # Cool down to half peak
                return self.initial_lr + (self.peak_lr - self.initial_lr) * (self.warmup_epochs / epoch)
            else:
                # After the warmup phase, allow ReduceLROnPlateau to manage the LR.
                after_warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr)/2
                # Slowly increase the learning rate after the warmup phase
                lr += lr/epoch
                return min(after_warmup_lr, lr)

    # Calculate class weights for imbalanced behavior data
    y_tr = model_dict['y_tr']
    if len(np.unique(y_tr)) == 2:  # Binary classification
        neg_count = np.sum(y_tr == 0)
        pos_count = np.sum(y_tr == 1)
        class_weight = {0: 1.0, 1: neg_count / max(pos_count, 1)}  # Inverse frequency weighting
        logger.info(f"📈 Class weighting - Negative: {neg_count}, Positive: {pos_count}, Weight: {class_weight[1]:.2f}")
    else:
        class_weight = None

    # Callbacks for training process control and monitoring
    
    # 1. Learning Rate Scheduler
    lr_schedule = CustomLRScheduler(initial_lr, peak_lr, warmup_epochs)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 2. Early Stopping with AUC monitoring for imbalanced data
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # 3. TensorBoard Callback with enhanced logging
    log_dir = save_folder / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )

    # 4. Model Checkpoint
    checkpoint_path = save_folder / "checkpoints" / f"{model_name}_best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0,
        save_weights_only=False
    )

    # 5. ReduceLROnPlateau
    reduce_lr_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=max(patience // 2, 2),
        verbose=1,
        min_lr=1e-7
    )

    # List of all callbacks
    callbacks = [
        lr_scheduler,
        early_stopping,
        tensorboard_callback,
        model_checkpoint,
        reduce_lr_on_plateau
    ]

    # Compile model with gradient clipping for stability
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_lr,
        clipnorm=1.0  # Gradient clipping to prevent explosions
    )
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Start training
    logger.info(f"🎯 Training configuration: {total_epochs} epochs, batch_size={batch_size}, lr=[{initial_lr}->{peak_lr}]")
    
    history = model.fit(
        X_tr, y_tr,
        epochs=total_epochs,
        batch_size=batch_size,
        validation_data=(model_dict['X_val_wide'], model_dict['y_val']),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    logger.info(f"✅ Training for model '{model_name}' completed.")
    logger.info(f"📊 Best validation loss: {min(history.history['val_loss']):.4f}")
    logger.info(f"📊 Best validation AUC: {max(history.history.get('val_auc', [0])):.4f}")
    
    return history

# %% Model Management Functions

def save_model(params_path: Path, model: tf.keras.Model, model_name: str) -> None:
    """
    Save a trained TensorFlow model.

    Args:
        modeling_path (Path): Path to modeling.yaml to get the save folder.
        model (tf.keras.Model): The trained Keras model to save.
        model_name (str): Name for the saved model file.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    save_folder = Path(modeling.get("models_path")) / 'trained_models'
    save_folder.mkdir(parents=True, exist_ok=True)
    
    filepath = save_folder / f"{model_name}.keras"
    model.save(filepath)
    logger.info(f"✅ Model '{model_name}' saved to: {filepath}")
    print(f"Model '{model_name}' saved to: {filepath}")

