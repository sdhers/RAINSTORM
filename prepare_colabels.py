"""
Created on Tue Mar 26 14:39:52 2024

@author: usuario
"""

import os
import pandas as pd

# At home:
desktop = 'C:/Users/dhers/Desktop/'

# At the lab:
# desktop = '/home/usuario/Desktop/'

folder = 'Labeling Santi Dhers'
path = desktop + folder
  
# Find all files ending with labels.csv
position_files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith('position.csv')]    

if not position_files:
    print("No files found in the specified directory.")

# Read each file into a DataFrame, sort by index, and concatenate
data = pd.DataFrame()

for file in position_files:
    position = pd.read_csv(file)
    labels = pd.read_csv(file.replace('_position.csv', '_labels.csv'))
    
    df = position.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    df['Left'] = labels['Left'] 
    df['Right'] = labels['Right']
    
    data = pd.concat([data, df], ignore_index = True)

# We remove the rows where the mice is not on the video
data = data.dropna(how='any')

X = data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                'neck_x', 'neck_y', 'body_x', 'body_y']].values

# Extract labels (exploring or not)
y = data[['Left', 'Right']].values

back = 1
forward = 1

reshaped_X = []

for i in range(back, len(X) - forward):
    reshaped_X.append(data[i - back : i + forward + 1])

#%%

def filter_dataframes(dataframes):
    # Create a new list to store filtered DataFrames
    filtered_dataframes = [df for df in dataframes if df.iloc[:, 0].nunique() != 2]
    
    # Calculate the number of removed rows
    removed_rows = len(dataframes) - len(filtered_dataframes)
    
    print(f"Removed {removed_rows} rows")
    
    return filtered_dataframes

#%%

X_wide = filter_dataframes(reshaped_X)

#%%

# reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)

#%%

"""
Created on Tue Mar 26 14:39:52 2024

@author: usuario
"""

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf

# At home:
desktop = 'C:/Users/dhers/Desktop/'

# At the lab:
# desktop = '/home/usuario/Desktop/'

folder = 'Labeling Santi Dhers'
path = desktop + folder

#%%

def extract_videos(path):
    
    position_files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith('position.csv')]

    """ Test """
        
    # Select a random video you want to use to test the model
    video_test = random.randint(1, len(position_files))

    # Select position and labels for testing
    position_test = position_files.pop(video_test - 1)
    labels_test = position_test.replace('_position.csv', '_labels.csv')
    
    position_df = pd.read_csv(position_test)
    labels_df = pd.read_csv(labels_test)
    
    test_data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    test_data['Left'] = labels_df['Left'] 
    test_data['Right'] = labels_df['Right']       
    
    # We remove the rows where the mice is not on the video
    test_data = test_data.dropna(how='any')
        
    X_test = test_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                    'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                    'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                    'neck_x', 'neck_y', 'body_x', 'body_y']].values
    
    # Extract labels (exploring or not)
    y_test = test_data[['Left', 'Right']].values
    
    
    """ val """
    
    # Select a random video you want to use to val the model
    video_val = random.randint(1, len(position_files))

    # Select position and labels for valing
    position_val = position_files.pop(video_val - 1)
    labels_val = position_val.replace('_position.csv', '_labels.csv')
    
    position_df = pd.read_csv(position_val)
    labels_df = pd.read_csv(labels_val)
    
    val_data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    val_data['Left'] = labels_df['Left'] 
    val_data['Right'] = labels_df['Right']       
    
    # We remove the rows where the mice is not on the video
    val_data = val_data.dropna(how='any')
        
    X_val = val_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                    'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                    'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                    'neck_x', 'neck_y', 'body_x', 'body_y']].values
    
    # Extract labels (exploring or not)
    y_val = val_data[['Left', 'Right']].values
    
    
    """ Train """
        
    train_data = pd.DataFrame()
    
    for file in range(len(position_files)):
        
        position_train = position_files[file]
        labels_train = position_train.replace('_position.csv', '_labels.csv')
    
        position_df = pd.read_csv(position_train)
        labels_df = pd.read_csv(labels_train)
        
        data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
        
        data['Left'] = labels_df['Left'] 
        data['Right'] = labels_df['Right']
    
        train_data = pd.concat([train_data, data], ignore_index = True)
    
    # We remove the rows where the mice is not on the video
    train_data = train_data.dropna(how='any')
        
    X_train = train_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                    'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                    'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                    'neck_x', 'neck_y', 'body_x', 'body_y']].values
    
    # Extract labels (exploring or not)
    y_train = train_data[['Left', 'Right']].values
    
    return X_test, y_test, X_val, y_val, X_train, y_train

#%%

X_test, y_test, X_val, y_val, X_train, y_train = extract_videos(path)

#%%

def filter_dataframes_1(dataframes):
    
    # Create a new list to store filtered DataFrames
    filtered_dataframes = [df for df in dataframes if df.iloc[:, 0].nunique() != 2]
    
    # Calculate the number of removed rows
    removed_rows = len(dataframes) - len(filtered_dataframes)
    
    print(f"Removed {removed_rows} rows")
    
    return filtered_dataframes

#%%

def filter_dataframes(dataframes):
    # Create a new list to store filtered DataFrames
    filtered_dataframes = []
    for df in dataframes:
        # Convert each 2D array slice into a DataFrame
        df = pd.DataFrame(df)
        # Check if the DataFrame has more than one unique value in the first column
        if df.iloc[:, 0].nunique() != 2:
            filtered_dataframes.append(df)
    
    # Calculate the number of removed rows
    removed_rows = len(dataframes) - len(filtered_dataframes)
    
    print(f"Removed {removed_rows} rows")
    
    return filtered_dataframes

#%%

def reshape_set(data, labels, back, forward):
    
    if labels is False:
        
        reshaped_data = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i - back : 1 + i + forward])
            
        filtered_data = filter_dataframes(reshaped_data)
        
        """
        reshaped_data_tf = tf.convert_to_tensor(filtered_data, dtype=tf.float64)
    
        return reshaped_data_tf
        """
        
        return filtered_data
        
    else:
        
        reshaped_data = []
        reshaped_labels = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i - back : 1 + i + forward])
            reshaped_labels.append(labels[i])
        
        filtered_data = filter_dataframes(reshaped_data)
        filtered_labels = filter_dataframes(reshaped_labels)
        
        """
        reshaped_data_tf = tf.convert_to_tensor(filtered_data, dtype=tf.float64)
        reshaped_labels_tf = tf.convert_to_tensor(filtered_labels, dtype=tf.float64)
    
        return reshaped_data_tf, reshaped_labels_tf
        """
        
        return filtered_data, filtered_labels

#%%

before = 1
after = 1

# Reshape the training set
X_train_seq, y_train_seq = reshape_set(X_train, y_train, before, after)

# Reshape the testing set
X_test_seq, y_test_seq = reshape_set(X_test, y_test, before, after)

# Reshape the validating set
X_val_seq, y_val_seq = reshape_set(X_val, y_val, before, after)