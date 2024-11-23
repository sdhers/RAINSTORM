
import os
import pandas as pd
import numpy as np

import plotly.graph_objects as go

import shutil
import random

from scipy import signal

# %% Functions for 1-Prepare_positions.ipynb

def choose_example(files: list, filter_word: str = 'TS') -> str:
    """Picks an example file from a list of files.

    Args:
        files (list): List of files to choose from.
        filter_word (str, optional): Word to filter files by. Defaults to 'TS'.

    Returns:
        str: Name of the chosen file.

    Raises:
        ValueError: If the files list is empty.
    """
    if not files:
        raise ValueError("The list of files is empty. Please provide a non-empty list.")

    filtered_files = [file for file in files if filter_word in file]

    if not filtered_files:
        print("No files found with the specified word")
        example = random.choice(files)
        print(f"Plotting coordinates from {os.path.basename(example)}")
    else:
        # Choose one file at random to use as example
        example = random.choice(filtered_files)
        print(f"Plotting coordinates from {os.path.basename(example)}")

    return example

def open_h5_file(path: str, print_data: bool = False, num_sd: float = 2) -> tuple:
    """Opens an h5 file and returns the data as a pandas dataframe.

    Args:
        path (str): Path to the h5 file.
        print_data (bool, optional): Whether to print the data. Defaults to False.
        num_sd (float, optional): Number of std_dev away from the mean. Defaults to 2.

    Returns:
        tuple: A tuple containing the dataframe and a list of bodyparts.
    """
    
    df = pd.read_hdf(path)
    scorer = df.columns.levels[0][0]
    bodyparts = df.columns.levels[1].to_list()
    df = df[scorer]

    df_raw = pd.DataFrame()

    for key in df.keys():
        df_raw[str(key[0]) + "_" + str(key[1])] = df[key]

    if print_data:
        print(f"Positions obtained by model: {scorer}")
        print(f"Points in df: {bodyparts}")
        for point in bodyparts:
            median = df_raw[f'{point}_likelihood'].median()
            mean = df_raw[f'{point}_likelihood'].mean()
            std_dev = df_raw[f'{point}_likelihood'].std()
            print(f'{point} \t median: {median:.2f} \t mean: {mean:.2f} \t std_dev: {std_dev:.2f} \t tolerance: {mean - num_sd*std_dev:.2f}')

    return df_raw, bodyparts

def filter_and_smooth_df(data: pd.DataFrame, bodyparts: list, objects: list, med_filt_window: int = 3, drop_below: float = 0.5, num_sd: float = 2) -> pd.DataFrame:
    """Filters and smooths a DataFrame of coordinates.

    Args:
        data (pd.DataFrame): DataFrame of coordinates.
        bodyparts (list): List of bodyparts to filter.
        objects (list): List of objects to filter.
        med_filt_window (int, optional): Window size for median filtering. Defaults to 3.
        drop_below (float, optional): Minimum likelihood to keep a bodypart. Defaults to 0.1.
        num_sd (float, optional): Number of standard deviations to use as the threshold. Defaults to 2.

    Returns:
        pd.DataFrame: Filtered and smoothed DataFrame of coordinates.
    """
    df = data.copy()

    # Try different filtering parameters
    sigma, n_sigmas = 0.6, 2
    N = int(2 * n_sigmas * sigma + 1)

    # Gaussian kernel
    gauss_kernel = signal.windows.gaussian(N, sigma)
    gauss_kernel = gauss_kernel / sum(gauss_kernel)
    pad_width = (len(gauss_kernel) - 1) // 2

    for point in bodyparts:

        median = df[f'{point}_likelihood'].median()
        mean = df[f'{point}_likelihood'].mean()
        std_dev = df[f'{point}_likelihood'].std()
            
        limit = mean - num_sd*std_dev

        # Set x and y coordinates to NaN where the likelihood is below the tolerance limit
        df.loc[df[f'{point}_likelihood'] < limit, [f'{point}_x', f'{point}_y']] = np.nan
        
        for axis in ['x','y']:
            column = f'{point}_{axis}'

            # Interpolate using the pchip method
            df[column] = df[column].interpolate(method='pchip', limit_area='inside')
            
            # Forward fill the remaining NaN values
            df[column] = df[column].ffill() #.bfill()
            
            # Apply median filter
            df[column] = signal.medfilt(df[column], kernel_size = med_filt_window)
            
            # Pad the median filtered data to mitigate edge effects
            padded = np.pad(df[column], pad_width, mode='edge')
            
            # Apply convolution
            smooth = signal.convolve(padded, gauss_kernel, mode='valid')
            
            # Trim the padded edges to restore original length
            df[column] = smooth[:len(df[column])]

            for obj in objects:
                if obj in column:
                    df[column] = df[column].median()

        # If the likelihood of an object is too low, probably the object is not there. Lets drop those columns
        if median < drop_below:
            df.drop([f'{point}_x', f'{point}_y', f'{point}_likelihood'], axis=1, inplace=True)
        
    return df

def plot_raw_vs_smoothed(df_raw, df_smooth, bodypart = 'nose', num_sd = 2):

    # Create figure
    fig = go.Figure()

    # Add traces for raw data
    for column in df_raw.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], mode='markers', name=f'raw {column}', marker=dict(symbol='x', size=6)))
            elif '_y' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], name=f'{column}', line=dict(color='black', width=3), yaxis='y2',opacity=0.5))

    # Add traces for smoothed data
    for column in df_smooth.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_smooth.index, y=df_smooth[column], name=f'new {column}', line=dict(width=3)))

    median = df_raw[f'{bodypart}_likelihood'].median()
    mean = df_raw[f'{bodypart}_likelihood'].mean()
    std_dev = df_raw[f'{bodypart}_likelihood'].std()
        
    limit = mean - num_sd*std_dev

    # Update layout for secondary y-axis
    fig.update_layout(
        xaxis=dict(title='Video frame'),
        yaxis=dict(title=f'{bodypart} position (pixels)'),
        yaxis2=dict(title=f'{bodypart} likelihood', 
                    overlaying='y', 
                    side='right',
                    gridcolor='black'),
        title=f'{bodypart} position & likelihood',
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h"),
        shapes=[dict(type='line', 
                    x0=df_raw.index.min(), 
                    x1=df_raw.index.max(), 
                    y0=limit, 
                    y1=limit, 
                    line=dict(color='green', dash='dash'),
                    yref='y2')],
    )

    # Show plot
    fig.show()

def find_scale_factor(df: pd.DataFrame, measured_dist: float, measured_points: list, print_results: bool = False) -> float:
    """Plots the distance between ears and the mean and median distances.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates of the points.
        measured_dist (float): Measured distance between points.
        measured_points (list): List of strings containing the names of the points.
        print_results (bool, optional): Whether to print the results. Defaults to False.

    Returns:
        scale (float): The scale factor to convert the measured distance to the actual distance.
    """

    df.dropna(inplace=True)

    A = measured_points[0]
    B = measured_points[1]

    # Calculate the distance between the two points
    dist = np.sqrt(
        (df[f'{A}_x'] - df[f'{B}_x'])**2 + 
        (df[f'{A}_y'] - df[f'{B}_y'])**2)

    dist.dropna(inplace=True)

    # Calculate the mean and median
    mean_dist = np.mean(dist)
    median_dist = np.median(dist)

    scale = (measured_dist / median_dist)

    if print_results:

        print(f'median distance is {median_dist}, mean distance is {mean_dist}. Scale factor is {scale:.4f} (1 cm = {1/scale:.2f} px).')

        # Create the plot
        fig = go.Figure()

        # Add the distance trace
        fig.add_trace(go.Scatter(y=dist, mode='lines', name='Distance between ears'))

        # Add mean and median lines
        fig.add_trace(go.Scatter(y=[mean_dist]*len(dist), mode='lines', name=f'Mean: {mean_dist:.2f}', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(y=[median_dist]*len(dist), mode='lines', name=f'Median: {median_dist:.2f}', line=dict(color='black')))

        # Update layout
        fig.update_layout(
            title=f'Distance between {A} and {B}',
            xaxis_title='Frame',
            yaxis_title='Distance (pixels)',
            legend=dict(yanchor="bottom",
                        y=1,
                        xanchor="center",
                        x=0.5,
                        orientation="h"),
        )

        # Show the plot
        fig.show()
    
    return scale

def process_position_files(files: list, objects: list = [], measured_dist: float = 1.8, measured_points: list = ['L_ear', 'R_ear'], scale: bool = True, fps: int = 30, med_filt_window: int = 3, drop_below: float = 0.5, num_sd: float = 2):
    """Processes a list of HDF5 files and saves the smoothed data as a CSV file.

    Args:
        files (list): List of HDF5 files to process.
        objects (list): List of objects to process.
        measured_dist (float): Measured distance between points in cm.
        measured_points (list): List of reference points for the distance calculation.
        scale (bool): Whether to scale the data.
        fps (int): Frames per second of the video.
        drop_below (float): Drop values if the median likelihood is below this threshold.
        num_sd (float): Number of standard deviations to consider.
    """
    
    for h5_file in files:

        df_raw, bodyparts = open_h5_file(h5_file)

        df_smooth = filter_and_smooth_df(df_raw, bodyparts, objects, med_filt_window, drop_below, num_sd)

        # Drop the likelihood columns
        df_smooth = df_smooth.drop(columns=df_smooth.filter(like='likelihood').columns)

        # Drop the frames when the mouse is not in the video
        df_smooth.dropna(inplace=True)
        
        if scale:
            # Use a constant that can be measured in real life to scale different sized videos from px to cm
            scale = find_scale_factor(df_smooth, measured_dist, measured_points)
            df_smooth = df_smooth * scale

        # Determine the output file path in the same directory as the input file
        # Split the path and filename
        input_dir, input_filename = os.path.split(h5_file)
        
        # Remove the original extension
        filename_without_extension = os.path.splitext(input_filename)[0]
        
        # Add the new extension '.csv'
        output_csv_path = os.path.join(input_dir, filename_without_extension + '.csv')
    
        # Save the processed data as a CSV file
        df_smooth.to_csv(output_csv_path, index=False)
        
        # Calculate the moment when the mouse enters the video
        mouse_enters = (len(df_raw) - len(df_smooth)) / fps

        print(f"{input_filename} has {df_smooth.shape[1]} columns. The mouse took {mouse_enters:.2f} sec to enter. Scale factor is {scale:.4f} (1 cm = {1/scale:.2f} px).")

def filter_and_move_files(folder: str, subfolders: list):
    """Filters and moves files to a subfolder.
    """
    for subfolder in subfolders:
        # Create a new subfolder
        output_folder = os.path.join(folder, subfolder, "position")
        os.makedirs(output_folder, exist_ok=True)

        # Get a list of all files in the input folder
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        # Iterate through files, move those without the word "position" to the "extra" subfolder
        for file in files:
            if subfolder in file and ".csv" in file and "filtered" not in file:
                file_path = os.path.join(folder, file)
                output_path = os.path.join(output_folder, file)

                # Move the file to the "extra" subfolder
                shutil.move(file_path, output_path)

    print("Files filtered and moved successfully.")

    """
    It also cleans all other files in the folder into a subfolder
    """
    subfolder = os.path.join(folder, "h5 files & others")
    os.makedirs(subfolder, exist_ok=True)
        
    # Get a list of all files in the input folder
    other_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    # Iterate through files, move those without the word "position" to the "extra" subfolder
    for file in other_files:
        file_path = os.path.join(folder, file)
        output_path = os.path.join(subfolder, file)
        
        # Move the file to the "extra" subfolder
        shutil.move(file_path, output_path)

    print("All .H5 files are stored away")

# %% Functions for 2-Geometric_analysis.ipynb

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

# %% Functions for 3a-Create_models.ipynb

# %%