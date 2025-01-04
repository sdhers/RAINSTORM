# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 2-Geometric_analysis.ipynb

# %% imports

import os
import pandas as pd
import numpy as np

import plotly.graph_objects as go

import random

from .utils import choose_example

# %% Functions

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

def plot_position(file: str, objects: list, maxDistance: float = 2.5, maxAngle: float = 45) -> None:
    """Plot mouse exploration around multiple objects.

    Args:
        file (str): Path to the .csv file containing the data.
        objects (list): List of objects to explore.
        maxDistance (float, optional): Maximum distance from the nose to the object. Defaults to 2.5.
        maxAngle (float, optional): Maximum angle between the head-nose and head-object vectors. Defaults to 45.
    """

    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'black', 'grey']
    symbol_list = ['square', 'circle', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
    
    # Read the .csv
    df = pd.read_csv(file)

    # Extract body parts
    nose = Point(df, 'nose')
    head = Point(df, 'head')

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

    # Loop over each object in the list of object names
    for idx, obj in enumerate(objects):

        # Create a Point object for the object
        obj_coords = Point(df, obj)

        # Find distance from the nose to the object
        dist = Point.dist(nose, obj_coords)
        
        # Compute the normalized head-object vector
        head_nose = Vector(head, nose, normalize=True)
        head_obj = Vector(head, obj_coords, normalize=True)
        
        # Find the angle between the head-nose and head-object vectors
        angle = Vector.angle(head_nose, head_obj)  # in degrees
        
        # Filter nose positions oriented towards the object
        towards_obj = nose.positions[(angle < maxAngle) & (dist < maxDistance**2)]
        
        # Create trace for filtered points oriented towards the object
        towards_trace = go.Scatter(
            x=towards_obj[:, 0],
            y=towards_obj[:, 1],
            mode='markers',
            marker=dict(opacity=0.4),
            name=f'Towards {obj}'
        )

        # Assign colors and symbols dynamically based on index
        object_color = color_list[idx]
        object_symbol = symbol_list[idx]

        # Create trace for the object
        obj_trace = go.Scatter(
            x=[obj_coords.positions[0][0]],
            y=[obj_coords.positions[0][1]],
            mode='markers',
            marker=dict(symbol=object_symbol, size=20, color=object_color),
            name=f'{obj}'
        )

        # Create circle around the object
        circle_trace = go.Scatter(
            x=obj_coords.positions[0][0] + maxDistance * np.cos(np.linspace(0, 2 * np.pi, 100)),
            y=obj_coords.positions[0][1] + maxDistance * np.sin(np.linspace(0, 2 * np.pi, 100)),
            mode='lines',
            line=dict(color='green', dash='dash'),
            name=f'{obj} radius'
        )

        # Append object-specific traces
        traces.append(towards_trace)
        traces.append(obj_trace)
        traces.append(circle_trace)

    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(file))[0]

    # Create layout
    layout = go.Layout(
        title=f'Object exploration in {filename}', 
        xaxis=dict(title='Horizontal position (cm)', scaleanchor='y'),  # Lock aspect ratio to the y-axis
        yaxis=dict(title='Vertical position (cm)'),
    )
    
    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Show plot
    fig.show()

def plot_freezing(file: str, fps: int = 30, threshold: float = 0.01) -> None:
    """Plots freezing events in a video.

    Args:
        file (str): Path to the video file.
        fps (int, optional): Frames per second of the video. Defaults to 30.
        threshold (float, optional): Threshold for freezing events. Defaults to 0.01.
    """
    
    # Load the CSV
    df = pd.read_csv(file)

    # Filter the position columns and exclude 'tail'
    position = df.filter(regex='_x|_y').filter(regex='^(?!.*tail_2)').filter(regex='^(?!.*tail_3)').copy()

    # Calculate movement based on the standard deviation of the difference in positions over a rolling window
    video_fps = 30  # Adjust as necessary
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

def create_movement_and_geolabels(files: list, objects: list, maxDistance: float = 2.5, maxAngle: float = 45, fps: int = 30, freezing_thr: float = 0.01, darting_thr: float = 0.8) -> None:
    """Analyzes the position data of a list of files and saves the results to a CSV file.

    Args:
        files (list): List of files to analyze.
        objects (list): List of objects to analyze.
        maxDistance (float, optional): Maximum distance between the mouse and the object. Defaults to 2.5.
        maxAngle (float, optional): Maximum angle between the mouse and the object. Defaults to 45.
        fps (int, optional): Frames per second of the video. Defaults to 30.
        freezing_thr (float, optional): Threshold for freezing. Defaults to 0.01.
        darting_thr (float, optional): Threshold for darting. Defaults to 0.8.
    """

    for file in files:
        
        # Determine the output file path
        input_dir, input_filename = os.path.split(file)
        parent_dir = os.path.dirname(input_dir)
        
        # Read the file
        position = pd.read_csv(file)

        if len(objects) != 0:

            # Initialize geolabels dataframe with columns for each object
            geolabels = pd.DataFrame(np.zeros((position.shape[0], len(objects))), columns=objects) 

            # Extract body parts
            nose = Point(position, 'nose')
            head = Point(position, 'head')

            # Check if all required object columns exist
            missing_objects = []

            for obj in objects:
                if f'{obj}_x' not in position.columns or f'{obj}_y' not in position.columns:
                    missing_objects.append(obj)
                    continue

                else:
                    # Extract the object's coordinates from the DataFrame
                    obj_coords = Point(position, obj)
                    
                    # Calculate the distance and angle between nose and the object
                    dist = Point.dist(nose, obj_coords)
                    head_nose = Vector(head, nose, normalize=True)
                    head_obj = Vector(head, obj_coords, normalize=True)
                    angle = Vector.angle(head_nose, head_obj)

                    # Loop over each frame and assign the geolabel if the mouse is exploring the object
                    for i in range(position.shape[0]):
                        if dist[i] < maxDistance and angle[i] < maxAngle:
                            geolabels.loc[i, obj] = 1  # Assign 1 if exploring the object                
            
            if len(missing_objects) != 0: # if true, there are no objects to analyze
                print(f"{input_filename} is missing objects: {', '.join(missing_objects)}")

            if len(objects) != len(missing_objects):
                # Convert geolabels to integer type (0 or 1)
                geolabels = geolabels.astype(int)

                # Insert a new column with the frame number at the beginning of the DataFrame
                geolabels.insert(0, "Frame", geolabels.index + 1)

                # Replace NaN values with 0
                geolabels.fillna(0, inplace=True)

                # Create a filename for the output CSV file
                output_filename_geolabels = input_filename.replace('_position.csv', '_geolabels.csv')
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
        movement = pd.DataFrame(np.zeros((position.shape[0], 4)), columns=["nose_dist", "body_dist", "freezing", "darting"])
        
        # Calculate the Euclidean distance between consecutive nose positions
        movement['nose_dist'] = (((position['nose_x'].diff())**2 + (position['nose_y'].diff())**2)**0.5) / 100
        movement['body_dist'] = (((position['body_x'].diff())**2 + (position['body_y'].diff())**2)**0.5) / 100
        movement['freezing'] = pd.DataFrame(np.where(moving_window < freezing_thr, 1, 0))
        movement['darting'] = pd.DataFrame(np.where(moving_window > darting_thr, 1, 0))

        movement.loc[:2*fps,:] = 0 # the first two seconds, as the mouse just entered, we dont quantify the movement
        
        # Insert a new column with the frame number at the beginning of the DataFrame
        movement.insert(0, "Frame", movement.index + 1)

        # Replace NaN values with 0
        movement.fillna(0, inplace=True)
        
        output_filename_movement = input_filename.replace('_position.csv', '_movement.csv')
        output_folder_distances = os.path.join(parent_dir + '/movement')
        os.makedirs(output_folder_distances, exist_ok = True)
        output_path_distances = os.path.join(output_folder_distances, output_filename_movement)
        movement.to_csv(output_path_distances, index=False)
            
        print(f"Saved movement to {output_filename_movement}")