import shutil
import os

def copy_folder(folder_path, suffix="_copy"):
    """
    Makes a copy of a folder.

    Parameters:
    folder_path (str): Path to the original folder.
    suffix (str): Suffix to add to the copied folder's name. Default is "_copy".

    Returns:
    str: Path to the copied folder.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' does not exist or is not a directory.")

    # Get the parent directory and the original folder name
    parent_dir, original_folder_name = os.path.split(folder_path.rstrip("/\\"))

    # Define the new folder name with the suffix
    copied_folder_name = f"{original_folder_name}{suffix}"
    copied_folder_path = os.path.join(parent_dir, copied_folder_name)

    # Check if the folder already exists
    if os.path.exists(copied_folder_path):
        print(f"The folder '{copied_folder_path}' already exists.")
    else:
        # Copy the folder
        shutil.copytree(folder_path, copied_folder_path)
        print(f"Copied folder to '{copied_folder_path}'.")

    return copied_folder_path

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