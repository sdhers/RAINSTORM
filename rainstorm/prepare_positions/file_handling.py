"""
RAINSTORM - Prepare Positions - File Handling

This script contains functions for file system operations such as
backing up folders, renaming files, and organizing files into subfolders.
"""

# %% Imports
import logging
import shutil
import stat
from pathlib import Path

from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Helper functions
def handle_remove_readonly(func, path, exc):
    """
    Error handler for shutil.rmtree to remove read-only files on Windows.
    This is necessary because shutil.rmtree cannot remove read-only files by default.
    """
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except Exception as e:
        logger.error(f"Failed to change permissions or remove {path}: {e}")
        raise


# %% Core functions
def backup_folder(folder_path: Path, suffix: str = "_backup", overwrite: bool = False) -> Path:
    """
    Makes a backup copy of a folder.

    Parameters:
        folder_path (Path): Path to the original folder.
        suffix (str): Suffix to add to the copied folder's name.
        overwrite (bool): If True, will overwrite the existing backup folder.

    Returns:
        Path: Path to the copied folder.

    Raises:
        ValueError: If folder_path does not exist or is not a directory.
        Exception: If backup creation or existing backup removal fails.
    """
    if not folder_path.is_dir():
        logger.error(f"The path '{folder_path}' does not exist or is not a directory.")
        raise ValueError(f"The path '{folder_path}' does not exist or is not a directory.")

    copied_folder_path = folder_path.with_name(f"{folder_path.name}{suffix}")

    if copied_folder_path.exists():
        if overwrite:
            print(f"Overwriting existing backup folder: '{copied_folder_path}'...")
            try:
                shutil.rmtree(copied_folder_path, onerror=handle_remove_readonly)
                logger.warning(f"Overwriting existing folder: '{copied_folder_path}'")
            except Exception as e:
                logger.error(f"Failed to remove existing backup folder '{copied_folder_path}': {e}")
                raise
        else:
            print(f"Backup folder already exists: '{copied_folder_path}'. Skipping backup.")
            logger.warning(f"The folder '{copied_folder_path}' already exists. Use overwrite=True to replace it.")
            return copied_folder_path

    print(f"Creating backup of '{folder_path}' to '{copied_folder_path}'...")
    try:
        shutil.copytree(folder_path, copied_folder_path)
        logger.info(f"Backup created at '{copied_folder_path}'")
    except Exception as e:
        logger.error(f"Failed to create backup of '{folder_path}' to '{copied_folder_path}': {e}")
        raise
    return copied_folder_path


def rename_files(folder_path: Path, old_substring: str, new_substring: str):
    """
    Renames files in a folder, replacing 'old_substring' with 'new_substring' in file names.

    Parameters:
        folder_path (Path): Path to the folder containing the files.
        old_substring (str): The substring to be replaced.
        new_substring (str): The substring to replace with.

    Raises:
        ValueError: If folder_path is not a valid directory.
    """
    if not folder_path.is_dir():
        logger.error(f"'{folder_path}' is not a valid directory.")
        raise ValueError(f"'{folder_path}' is not a valid directory.")

    modified_any_file = False
    print(f"Attempting to rename files in '{folder_path}'...")

    for old_file_path in folder_path.iterdir():
        if old_file_path.is_file() and old_substring in old_file_path.name:
            new_file_name = old_file_path.name.replace(old_substring, new_substring)
            new_file_path = old_file_path.with_name(new_file_name)

            if new_file_path.exists():
                logger.warning(f"Skipping rename of '{old_file_path.name}' to '{new_file_name}' as target file already exists.")
                print(f"Skipping: '{old_file_path.name}' to '{new_file_name}' (target exists).")
                continue

            try:
                old_file_path.rename(new_file_path)
                logger.info(f"Renamed: '{old_file_path.name}' → '{new_file_name}'")
                print(f"Renamed: '{old_file_path.name}' → '{new_file_name}'")
                modified_any_file = True
            except OSError as e:
                logger.error(f"Failed to rename '{old_file_path.name}' to '{new_file_name}': {e}")
                print(f"Error renaming '{old_file_path.name}': {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while renaming '{old_file_path.name}': {e}")
                print(f"Unexpected error renaming '{old_file_path.name}': {e}")

    if not modified_any_file:
        print(f"No files found containing '{old_substring}' in '{folder_path}' to rename.")
        logger.info(f"No files modified in '{folder_path}' with '{old_substring}'.")
    else:
        print("File renaming process complete.")


def filter_and_move_files(params_path: Path, trials_subfolder: str = "positions", h5_subfolder: str = "h5_files"):
    """
    Filters CSVs by trial name into per-trial subfolders, and moves all .h5 files into an archive subfolder.

    Args:
        params_path (Path): Path to the YAML parameters file.
        trials_subfolder (str): Name of the subfolder under each trial to store CSVs.
        h5_subfolder (str): Name of the subfolder under the main folder to store .h5 files.
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    trials = params.get("seize_labels", {}).get("trials", [])

    if not folder_path.is_dir():
        logger.error(f"Invalid folder path in params: '{folder_path}'")
        print(f"Error: Invalid folder path in params: '{folder_path}'.")
        return {}

    # Move CSVs into trial-specific subfolders
    for trial in trials:
        dest_dir = folder_path / trial / trials_subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensuring directory exists: '{dest_dir}'")

        for fname in folder_path.iterdir():
            if fname.is_file() and fname.suffix.lower() == ".csv" and trial in fname.name:
                src = fname
                dst = dest_dir / fname.name
                try:
                    shutil.move(src, dst)
                    logger.info(f"Moved CSV '{fname.name}' → '{trial}/{trials_subfolder}/'")
                    print(f"Moved CSV '{fname.name}' → '{trial}/{trials_subfolder}/'")
                except Exception as e:
                    logger.error(f"Failed to move '{src}' → '{dst}': {e}")
                    print(f"Error moving '{src}': {e}")

    # Archive all remaining .h5 files
    h5_archive_dir = folder_path / h5_subfolder
    h5_archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensuring H5 archive directory exists: '{h5_archive_dir}'")

    for fname in folder_path.iterdir():
        if fname.is_file() and fname.suffix.lower() == ".h5":
            src = fname
            dst = h5_archive_dir / fname.name
            try:
                shutil.move(src, dst)
                logger.info(f"Archived H5 '{fname.name}' → '{h5_subfolder}/'")
                print(f"Archived H5 '{fname.name}' → '{h5_subfolder}/'")
            except Exception as e:
                logger.error(f"Failed to archive '{src}' → '{dst}': {e}")
                print(f"Error archiving '{src}': {e}")

    print("File filtering and moving complete.")
    logger.info("File filtering and moving complete.")
