"""
RAINSTORM - Modeling - Create Colabels

This script processes position data and behavior labels from multiple labelers
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List

from .utils import configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Functions

def create_colabels(data_dir: Path, labelers: List[str], targets: List[str]) -> None:
    """
    Create a combined dataset (colabels) with mouse position data, object positions,
    and behavior labels from multiple labelers.

    Args:
        data_dir (Path): Path to the directory containing the 'positions' folder and labeler folders.
        labelers (List[str]): Folder names for each labeler, relative to `data_dir`.
        targets (List[str]): Names of the stationary exploration targets.

    Output:
        Saves a 'colabels.csv' file in the `data_dir`.
    """
    position_dir = data_dir / 'positions'
    if not position_dir.is_dir():
        logger.error(f"'positions' folder not found in {data_dir}")
        return
    
    logger.info(f"📂 Processing position files in: {position_dir}")
    position_files = [f for f in position_dir.iterdir() if f.suffix == '.csv']
    if not position_files:
        logger.error(f"No position files found in {position_dir}")
        return

    all_entries = []

    for filename in position_files:
        pos_df = pd.read_csv(filename)

        # Identify body part columns by excluding all target-related columns
        bodypart_cols = [col for col in pos_df.columns if not any(col.startswith(f'{tgt}') for tgt in targets)]
        bodyparts_df = pos_df[bodypart_cols]

        for tgt in targets:
            if f'{tgt}_x' not in pos_df.columns or f'{tgt}_y' not in pos_df.columns:
                logger.error(f"Missing coordinates for target '{tgt}' in {filename.name}")
                return

            target_df = pos_df[[f'{tgt}_x', f'{tgt}_y']].rename(columns={f'{tgt}_x': 'obj_x', f'{tgt}_y': 'obj_y'})

            # Load label data from each labeler
            label_data = {}
            for labeler in labelers:
                label_file = data_dir / labeler / filename.name.replace('_position.csv', '_labels.csv')
                if not label_file.is_file():
                    logger.error(f"Label file missing: {label_file}")
                    return
                
                label_df = pd.read_csv(label_file)
                if tgt not in label_df.columns:
                    logger.error(f"Label column '{tgt}' not found in {label_file.name}")
                    return
                
                label_data[labeler] = label_df[tgt]

            # Combine everything into one DataFrame
            combined_df = pd.concat(
                [bodyparts_df, target_df] + [label_data[labeler].rename(labeler) for labeler in labelers],
                axis=1
            )
            all_entries.append(combined_df)

    # Final DataFrame
    colabels_df = pd.concat(all_entries, ignore_index=True)

    # Save to CSV
    output_path = data_dir / 'colabels.csv'
    colabels_df.to_csv(output_path, index=False)
    logger.info(f"Colabels saved to: {output_path}")
    print(f"Colabels saved to: {output_path}")