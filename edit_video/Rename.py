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

folder_path = r'C:\Users\dhers\OneDrive - UBA\NOR position\3xTg\1st batch\2024-06_Tg-9m'

before = 'DLC_resnet50_STORMJul11shuffle1_100000'

after = '_position'

rename_files(folder_path, before, after)

#%%

folder_path = r'C:\Users\dhers\Desktop\2023-11_Tg-2m'

before = 'C4_B_L'

after = 'R11_C06-a_B_L'
 
rename_files(folder_path, before, after)