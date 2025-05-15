# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 2-Geometric_analysis.ipynb

# %% imports

import os
import pandas as pd
import numpy as np
import yaml
from glob import glob
import random
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %% Functions

def load_yaml(params_path: str) -> dict:
    """Loads a YAML file."""
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

class Point:
    def __init__(self, df, table):

        x = df[table + '_x']
        y = df[table + '_y']

        self.positions = np.dstack((x, y))[0]

    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1.positions - p2.positions, axis=1)

class Vector:
    def __init__(self, p1, p2, normalize=True):

        self.positions = p2.positions - p1.positions

        self.norm = np.linalg.norm(self.positions, axis=1)

        if normalize:
            self.positions = self.positions / np.repeat(np.expand_dims(self.norm,axis=1), 2, axis=1)

    @staticmethod
    def angle(v1, v2):
        
        length = len(v1.positions)
        angle = np.zeros(length)

        for i in range(length):
            angle[i] = np.rad2deg(np.arccos(np.dot(v1.positions[i], v2.positions[i])))

        return angle

def choose_example_csv(params_path, look_for: str = 'TS') -> str:
    """Picks an example file from a list of files.

    Args:
        files (list): List of files to choose from.
        look_for (str, optional): Word to filter files by. Defaults to 'TS'.

    Returns:
        str: Name of the chosen file.

    Raises:
        ValueError: If the files list is empty.
    """
    params = load_yaml(params_path)
    folder_path = params.get("path")
    filenames = params.get("filenames")
    trials = params.get("seize_labels", {}).get("trials", [])
    files = []
    for trial in trials:
        temp_files = [os.path.join(folder_path, trial, 'positions', file + '_positions.csv') for file in filenames if trial in file]
        files.extend(temp_files)
    
    if not files:
        raise ValueError("The list of files is empty. Please provide a non-empty list.")

    filtered_files = [file for file in files if look_for in file]

    if not filtered_files:
        print("No files found with the specified word")
        example = random.choice(files)
        print(f"Plotting coordinates from {os.path.basename(example)}")
    else:
        # Choose one file at random to use as example
        example = random.choice(filtered_files)
        print(f"Plotting coordinates from {os.path.basename(example)}")

    return example

def plot_positions(params_path: str, file: str, scale: bool = True) -> None:
    """Plot mouse exploration around multiple targets.

    Args:
        params_path (str): Path to the YAML parameters file.
        file (str): Path to the .csv file containing the data.
    """
    # Load parameters
    params = load_yaml(params_path)
    targets = params.get("targets", [])

    # Load geometric analysis parameters
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    max_distance = geometric_params.get("distance", 2.5)
    orientation = geometric_params.get("orientation", {})
    max_angle = orientation.get("degree", 45)  # in degrees
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')

    # Define colors and symbols
    symbol_list = ['square', 'circle', 'diamond', 'cross', 'x']
    color_list =        ['blue',        'darkred',      'darkgreen',    'purple',   'goldenrod']
    trace_color_list =  ['turquoise',   'orangered',    'limegreen',    'magenta',  'gold']

    # Create a dictionary mapping targets to their properties
    target_styles = {
        tgt: {            
            "symbol": symbol_list[idx % len(symbol_list)],
            "color": color_list[idx % len(color_list)],  # Avoid index errors
            "trace_color": trace_color_list[idx % len(trace_color_list)]  # Darker shade for towards_trace
        }
        for idx, tgt in enumerate(targets)
    }

    # Read the .csv
    df = pd.read_csv(file)

    if scale:
        # Scale the data
        df *= 1 / scale

    # Extract body parts
    nose = Point(df, front)
    head = Point(df, pivot)

    # Create the main trace for nose positions
    nose_trace = go.Scatter(
        x=nose.positions[:, 0],
        y=nose.positions[:, 1],
        mode='markers',
        marker=dict(color='grey', opacity=0.2),
        name='Nose Positions'
    )

    # Store all traces
    traces = [nose_trace]

    # Loop over each target
    for tgt in targets:
        if f'{tgt}_x' in df.columns:
            # Get target properties from dictionary
            target_color = target_styles[tgt]["color"]
            target_symbol = target_styles[tgt]["symbol"]
            towards_trace_color = target_styles[tgt]["trace_color"]

            # Create a Point target for the target
            tgt_coords = Point(df, tgt)

            # Find distance from the nose to the target
            dist = Point.dist(nose, tgt_coords)

            # Compute the normalized head-target vector
            head_nose = Vector(head, nose, normalize=True)
            head_tgt = Vector(head, tgt_coords, normalize=True)

            # Find the angle between the head-nose and head-target vectors
            angle = Vector.angle(head_nose, head_tgt)  # in degrees

            # Filter nose positions oriented towards the target
            towards_tgt = nose.positions[(angle < max_angle) & (dist < max_distance * 3)]

            # Create trace for filtered points oriented towards the target
            towards_trace = go.Scatter(
                x=towards_tgt[:, 0],
                y=towards_tgt[:, 1],
                mode='markers',
                marker=dict(opacity=0.4, color=towards_trace_color),
                name=f'Towards {tgt}'
            )

            # Create trace for the target
            tgt_trace = go.Scatter(
                x=[tgt_coords.positions[0][0]],
                y=[tgt_coords.positions[0][1]],
                mode='markers',
                marker=dict(symbol=target_symbol, size=20, color=target_color),
                name=f'{tgt}'
            )

            # Create circle around the target
            circle_trace = go.Scatter(
                x=tgt_coords.positions[0][0] + max_distance * np.cos(np.linspace(0, 2 * np.pi, 100)),
                y=tgt_coords.positions[0][1] + max_distance * np.sin(np.linspace(0, 2 * np.pi, 100)),
                mode='lines',
                line=dict(color='green', dash='dash'),
                name=f'{tgt} radius'
            )

            # Append target-specific traces
            traces.append(towards_trace)
            traces.append(tgt_trace)
            traces.append(circle_trace)

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(file))[0]

    # Create layout
    layout = go.Layout(
        title=f'Target exploration in {filename}',
        xaxis=dict(title='Horizontal position (cm)', scaleanchor='y'),
        yaxis=dict(title='Vertical position (cm)', autorange="reversed")
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Show plot
    fig.show()

def point_in_roi(x, y, center, width, height, angle):
    """Check if a point (x, y) is inside a rotated rectangle."""
    angle = np.radians(angle)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Translate point relative to the rectangle center
    x_rel, y_rel = x - center[0], y - center[1]

    # Rotate the point in the opposite direction
    x_rot = x_rel * cos_a + y_rel * sin_a
    y_rot = -x_rel * sin_a + y_rel * cos_a

    # Check if the point falls within the unrotated rectangle's bounds
    return (-width / 2 <= x_rot <= width / 2) and (-height / 2 <= y_rot <= height / 2)

def detect_roi_activity(params_path, file, bodypart = 'body', plot_activity = False, verbose = True):

    """Assigns an area to each body part for each frame."""
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps", 30)
    areas = params.get("geometric_analysis", {}).get("roi_data", {}).get("areas", [])
    
    if not areas:
        if verbose:
            print("No ROIs found in the parameters file. Skipping ROI activity analysis.")
        return

    # Read the .csv
    df = pd.read_csv(file)

    # Create a new DataFrame for results
    roi_activity = pd.DataFrame(index=df.index)

    area_col = []
    for i, row in df.iterrows():
        x, y = row[f"{bodypart}_x"], row[f"{bodypart}_y"]
        assigned_area = 'other'
        for area in areas:
            if point_in_roi(x, y, area["center"], area["width"], area["height"], area["angle"]):
                assigned_area = area["name"]
                break
        area_col.append(assigned_area)
    
    roi_activity['location'] = area_col  # Assign the area to the corresponding body part column

    if plot_activity:

        """
        Plot the time spent in each area for a specific body part.
        """
        # Count occurrences of each area
        time_spent = roi_activity['location'].value_counts().sort_index()
        time_spent = time_spent[time_spent.index != 'other']

        # Convert frame count to time (seconds)
        time_spent_seconds = time_spent / fps

        # Plot
        plt.figure()
        time_spent_seconds.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.xlabel("Area")
        plt.ylabel("Time spent (s)")
        plt.title(f"Time spent in each area - {bodypart}")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
    
    return roi_activity

def plot_heatmap(params_path, file, bodypart = 'body', bins=50, cmap="coolwarm", alpha=0.75):
    """
    Plots a heatmap of body part positions overlaid with ROIs.

    Parameters:
    - params_path: Path to the parameters file.
    - file: Path to the .csv file containing the positions.
    - bodypart: The body part to analyze (e.g., "nose").
    - bins: Number of bins for the heatmap (default: 50).
    - cmap: Colormap for the heatmap (default: "Reds"). Other options: "inferno", "plasma", "cividis", "magma".
    - alpha: Transparency level for the heatmap (default: 0.6).
    """

    # Load parameters
    params = load_yaml(params_path)
    roi_data = params.get("geometric_analysis", {}).get("roi_data", {})
    areas = roi_data.get("areas", {})
    frame_shape = roi_data.get("frame_shape", [])
    frame_width = frame_shape[0]
    frame_height = frame_shape[1]

    if not frame_width or not frame_height:
        print("Frame shape not found in the parameters file. Skipping heatmap plot.")
        return

    # Read the .csv
    df = pd.read_csv(file)

    # Extract x and y positions of the body part
    x_vals = df[f"{bodypart}_x"].dropna().values
    y_vals = df[f"{bodypart}_y"].dropna().values

    # Create a 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins, 
                                             range=[[0, frame_width], [0, frame_height]])

    # Transpose the heatmap for correct orientation
    heatmap = heatmap.T

    # Create figure
    fig, ax = plt.subplots(figsize=(frame_width / 100, frame_height / 100))  # Scale size
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    ax.invert_yaxis()  # Invert Y-axis to match video coordinates
    ax.set_title(f"Heatmap of {bodypart} positions")
    ax.axis("off")

    # Plot heatmap
    ax.imshow(heatmap, extent=[0, frame_width, 0, frame_height], origin="lower", cmap=cmap, alpha=alpha)

    # Plot ROIs
    if areas:
        for area in areas:
            center_x, center_y = area["center"]
            width, height = area["width"], area["height"]
            angle = area["angle"]

            # Create rotated rectangle
            rect = patches.Rectangle(
                (center_x - width / 2, center_y - height / 2), width, height,
                angle=angle, rotation_point="center", edgecolor="black", facecolor="none", lw=2
            )
            ax.add_patch(rect)
            ax.text(center_x, center_y, area["name"], fontsize=10, color="black", 
                    ha="center", va="center", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.show()

def plot_freezing(params_path:str, file: str) -> None:
    """Plots freezing events in a video.

    Args:
        params_path (str): Path to the YAML parameters file.
        file (str): Path to the .csv file containing the data.
    """
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps", 30)
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    threshold = geometric_params.get("freezing_threshold", 0.01)
    
    # Load the CSV
    df = pd.read_csv(file)

    # Scale the data
    df *= 1/scale

    # Filter the position columns and exclude 'tail'
    position = df.filter(regex='_x|_y').filter(regex='^(?!.*tail_2)').filter(regex='^(?!.*tail_3)').copy()

    # Calculate movement based on the standard deviation of the difference in positions over a rolling window
    movement = position.diff().rolling(window=int(fps), center=True).std().mean(axis=1)

    # Create a DataFrame indicating freezing events (when movement is below 0.01)
    freezing = pd.DataFrame(np.where(movement < threshold, 1, 0), columns=['freezing'])

    # Create a time axis (frame number divided by frame rate gives time in seconds)
    time = np.arange(len(movement)) / fps

    # Create the plot using Plotly
    fig = go.Figure()

    # Add the movement line
    fig.add_trace(go.Scatter(x=time, y=movement, mode='lines', name='Movement'))

    # Identify the start and end of freezing events
    freezing['change'] = freezing['freezing'].diff()
    freezing['event_id'] = (freezing['change'] == 1).cumsum()  # Create groups of consecutive 1's

    # Filter for events where freezing is 1
    events = freezing[freezing['freezing'] == 1]

    # Iterate over each event and add shapes
    for event_id, group in events.groupby('event_id'):
        start_index = group.index[0]
        end_index = group.index[-1]
        # Add the rectangle from the start to the end of the event
        fig.add_shape(
            type='rect',
            x0=time[start_index], x1=time[end_index + 1],  # Adjust x1 to include the last time point
            y0=-0.1, y1=1.5,
            fillcolor='rgba(0, 100, 80, 0.4)',
            line=dict(width=0.4),
        )

    # Add a horizontal line for the freezing threshold
    fig.add_hline(y=threshold, line=dict(color='red', dash='dash'),
              annotation_text='Freezing Threshold', annotation_position='bottom left')

    # Customize the layout
    fig.update_layout(
        title=f'Freezing Events in {os.path.basename(file)}',
        xaxis=dict(title='Time (seconds)', range=[0, time.max() + 1]),  # Adjust x-axis
        yaxis=dict(title='General Movement (cm)', range=[-0.05, movement.max() * 1.05]),  # Adjust y-axis
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h"),
        showlegend=True,
    )

    # Show the plot
    fig.show()

def create_movement_and_geolabels(params_path:str, wait: int = 2, roi_bodypart = 'body') -> None:
    """Analyzes the position data of a list of files.

    Args:
        params_path (str): Path to the YAML parameters file.
        wait (int, optional): Number of seconds to wait before starting to measure movement. Defaults to 2.
    """
    params = load_yaml(params_path)
    folder_path = params.get("path")
    filenames = params.get("filenames")
    trials = params.get("seize_labels", {}).get("trials", [])
    files = []
    for trial in trials:
        temp_files = [os.path.join(folder_path, trial, 'positions', file + '_positions.csv') for file in filenames if trial in file]
        files.extend(temp_files)
    targets = params.get("targets", [])
    fps = params.get("fps", 30)
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    max_distance = geometric_params.get("distance", 2.5) # in cm
    orientation = geometric_params.get("orientation", {})
    max_angle = orientation.get("degree", 45)  # in degrees
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')
    freezing_threshold = geometric_params.get("freezing_threshold", 0.01)

    for file in files:
        
        # Determine the output file path
        input_dir, input_filename = os.path.split(file)
        parent_dir = os.path.dirname(input_dir)
        
        # Read the file
        position = pd.read_csv(file)

        # Scale the data
        position *= 1/scale

        if targets:

            # Initialize geolabels dataframe with columns for each target
            geolabels = pd.DataFrame(np.zeros((position.shape[0], len(targets))), columns=targets) 

            # Extract body parts
            nose = Point(position, front)
            head = Point(position, pivot)

            # Check if all required target columns exist
            missing_targets = []

            for obj in targets:
                if f'{obj}_x' not in position.columns or f'{obj}_y' not in position.columns:
                    missing_targets.append(obj)
                    continue

                else:
                    # Extract the target's coordinates from the DataFrame
                    obj_coords = Point(position, obj)
                    
                    # Calculate the distance and angle between nose and the target
                    dist = Point.dist(nose, obj_coords)
                    head_nose = Vector(head, nose, normalize=True)
                    head_obj = Vector(head, obj_coords, normalize=True)
                    angle = Vector.angle(head_nose, head_obj)

                    # Loop over each frame and assign the geolabel if the mouse is exploring the target
                    for i in range(position.shape[0]):
                        if dist[i] < max_distance and angle[i] < max_angle:
                            geolabels.loc[i, obj] = 1  # Assign 1 if exploring the target                
            
            if len(missing_targets) != 0: # if true, there are no targets to analyze
                print(f"{input_filename} is missing targets: {', '.join(missing_targets)}")

            if len(targets) != len(missing_targets):
                # Convert geolabels to integer type (0 or 1)
                geolabels = geolabels.astype(int)

                # Insert a new column with the frame number at the beginning of the DataFrame
                geolabels.insert(0, "Frame", geolabels.index + 1)

                # Replace NaN values with 0
                geolabels.fillna(0, inplace=True)

                # Create a filename for the output CSV file
                output_filename_geolabels = input_filename.replace('_positions.csv', '_geolabels.csv')
                output_folder_geolabels = os.path.join(parent_dir + '/geolabels')
                os.makedirs(output_folder_geolabels, exist_ok=True)
                output_path_geolabels = os.path.join(output_folder_geolabels, output_filename_geolabels)

                # Save geolabels to CSV
                geolabels.to_csv(output_path_geolabels, index=False)
                print(f"Saved geolabels to {output_filename_geolabels}")

        # Filter the position columns and exclude 'tail'
        tail_less = position.filter(regex='^(?!.*tail_2)').filter(regex='^(?!.*tail_3)').copy()

        # Calculate movement based on the standard deviation of the difference in positions over a rolling window
        moving_window = tail_less.diff().rolling(window=int(fps), center=True).std().mean(axis=1)

        # Create the distances dataframe
        movement = pd.DataFrame(np.zeros((position.shape[0], 3)), columns=["nose_dist", "body_dist", "freezing"])

        # Calculate the Euclidean distance between consecutive nose positions
        movement['nose_dist'] = (((position['nose_x'].diff())**2 + (position['nose_y'].diff())**2)**0.5) / 100
        movement['body_dist'] = (((position['body_x'].diff())**2 + (position['body_y'].diff())**2)**0.5) / 100
        movement['freezing'] = pd.DataFrame(np.where(moving_window < freezing_threshold, 1, 0))

        movement.loc[:wait*fps,:] = 0 # the first two seconds, as the mouse just entered the arena, we dont quantify the movement

        # Calculate ROI activity
        roi_activity = detect_roi_activity(params_path, file, bodypart = roi_bodypart, plot_activity = False, verbose = False)

        movement = pd.concat([movement, roi_activity], axis = 1)
        
        # Insert a new column with the frame number at the beginning of the DataFrame
        movement.insert(0, "Frame", movement.index + 1)

        # Replace NaN values with 0
        movement.fillna(0, inplace=True)
        
        output_filename_movement = input_filename.replace('_positions.csv', '_movement.csv')
        output_folder_distances = os.path.join(parent_dir + '/movement')
        os.makedirs(output_folder_distances, exist_ok = True)
        output_path_distances = os.path.join(output_folder_distances, output_filename_movement)
        movement.to_csv(output_path_distances, index=False)
            
        print(f"Saved movement to {output_filename_movement}")