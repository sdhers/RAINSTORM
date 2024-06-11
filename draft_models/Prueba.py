# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:29:21 2024

@author: dhers
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold

#%%

# State your path
desktop = 'C:/Users/dhers/Desktop'

STORM_folder = os.path.join(desktop, 'STORM/models')
colabels_file = os.path.join(STORM_folder, 'colabeled_data.csv')
colabels = pd.read_csv(colabels_file)


# Set the number of neurons in each layer
param_0 = 16
param_H1 = 32
param_H2 = 64
param_H3 = 32
param_H4 = 16

batch_size = 128 # Set the batch size
initial_lr = 0.001 # Set the initial learning rate
epochs = 20 # Set the training epochs
patience = 5 # Set the wait for the early stopping mechanism

n_splits = 5  # Number of cross-validation splits

#%%

X = colabels.iloc[:, :16] # We leave out the tail

y = colabels.iloc[:, 30:32]

#%%

# Define a function to create a new instance of the model
def create_model():
    model = Sequential([
        Dense(param_0, input_shape=(X.shape[1],)),
        Dense(param_H1),
        Dense(param_H2),
        Dense(param_H3),
        Dense(param_H4),
        Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%%

# Wrap the model using KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

#%%

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#%%

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

#%%

# Print cross-validation results
print("Cross-validation scores: ", cv_scores)
print("Average cross-validation score: ", np.mean(cv_scores))
