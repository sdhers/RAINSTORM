"""
Created on Tue Mar 26 14:39:52 2024

@author: usuario
"""

import os
import pandas as pd

path = '/home/usuario/Desktop/Labeling Santi Dhers'
  
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
