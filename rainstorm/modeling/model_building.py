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
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# %% Model Building Functions

def build_RNN(modeling: Dict[str, Any], model_dict: Dict[str, np.ndarray]) -> tf.keras.Model:
    """
    Builds a Bidirectional LSTM (RNN) model with optional attention mechanism.

    Args:
        modeling (Dict[str, Any]): Dictionary containing modeling parameters,
                                   especially 'RNN' configuration.
        model_dict (Dict[str, np.ndarray]): Dictionary containing split data,
                                            e.g., 'X_tr' to determine input shape.

    Returns:
        tf.keras.Model: Compiled Keras RNN model.
    """
    rnn_conf = modeling.get("RNN", {})
    units = rnn_conf.get("units", [16, 24, 32, 24, 16, 8])
    dropout_rate = rnn_conf.get("dropout", 0.2)
    initial_lr = rnn_conf.get("initial_lr", 1e-5)

    input_shape = model_dict['X_tr_wide'].shape[1:]  # (timesteps, features)
    print(list(input_shape)) # For debugging/verification as in original notebook

    # Input layer
    input_sequence = Input(shape=input_shape, name="input_sequence")

    # RNN layers with Batch Normalization and Dropout
    x = input_sequence
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
        x = add_attn # The output of this block feeds into the next layer

        # Cropping (to simulate the temporal window shrinking if needed, as in original)
        # This part assumes a specific temporal window shrinking logic.
        # If the model is meant to maintain sequence length until final pooling, this might need adjustment.
        # For simplicity, if `broad` is not 1, it might have impacted the input shape already.
        # This Cropping1D typically removes one timestep from each end.
        if i < len(units) -1: # Apply cropping for intermediate layers
            x = Cropping1D(cropping=(1, 0), name=f"crop_{i}")(x) # Remove one timestep from the start

    # Attention Pooling layer (if final output needs to be a single vector)
    # Global attention pooling to convert sequence to a single vector
    attention_score = Dense(1, name="attention_pooling_score")(x)
    attention_weights = Activation('softmax', name="attention_pooling_weights")(attention_score)
    attention_output = Multiply(name="attention_pooling_weighted")([x, attention_weights])
    attention_pooled = Lambda(lambda z: tf.reduce_sum(z, axis=1), name="attention_pooling_sum")(attention_output)

    # Output layer
    binary_out = Dense(1, activation='sigmoid', name="binary_out")(attention_pooled)

    model = Model(inputs=input_sequence, outputs=binary_out, name="ModularBidirectionalRNN")

    # Compile the model
    # Use AdamW if available for better regularization, otherwise Adam.
    # Learning rate is set by LearningRateScheduler, so this is just a default.
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

