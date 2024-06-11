# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:30:38 2024

@author: dhers
"""

import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Enable device placement logging
tf.debugging.set_log_device_placement(True)

# Define a simple computation to see the device placement
a = tf.constant([[1.0, 2.0, 3.0]])
b = tf.constant([[4.0, 5.0, 6.0]])
c = tf.matmul(a, b, transpose_b=True)

print(c)

# Optionally, start a TensorFlow Profiler session
# from tensorflow.python.eager import profiler
# profiler.start()
# # Run some computations...
# profiler.stop()

# Print GPU device details
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU found.")
