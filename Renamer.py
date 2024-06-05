# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:06:23 2024

@author: dhers
"""

import os

def rename_files(folder, before, after):
    # Get a list of all files in the specified folder
    files = os.listdir(folder)
    
    for file_name in files:
        # Check if 'before' is in the file name
        if before in file_name:
            # Construct the new file name
            new_name = file_name.replace(before, after)
            # Construct full file paths
            old_file = os.path.join(folder, file_name)
            new_file = os.path.join(folder, new_name)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')

#%%

folder_path = r'C:\Users\dhers\OneDrive - UBA\NOR position\Pharma\2024-05_PD-45'

before = 'DLC_resnet50_VaderDec1shuffle1_200000'

after = '_position'

rename_files(folder_path, before, after)

#%%

folder_path = r'C:\Users\dhers\OneDrive - UBA\NOR position\Pharma\2024-05_PD-45'

before = 'TS2'

after = 'TW'

rename_files(folder_path, before, after)