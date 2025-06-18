# %% ROI activity

def plot_roi_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None, outliers=[]) -> None:
    """
    Plot the average time spent in each ROI area.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    folder = os.path.join(path, 'summary', group, trial)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    all_roi_times = {}

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        filename = os.path.basename(file_path)
        if any(outlier in filename for outlier in outliers):
            continue  # Skip files matching any outlier name
        df = pd.read_csv(file_path)

        # Count total time spent in each area (convert frames to seconds)
        roi_times = df.groupby('location').size() / fps

        # Store values in a dictionary
        for roi, time in roi_times.items():
            if roi not in all_roi_times:
                all_roi_times[roi] = []
            all_roi_times[roi].append(time)

    # Sort ROI names to keep plots consistent across groups
    roi_labels = sorted(all_roi_times.keys())  
    num_rois = len(roi_labels)
    space = 1/(num_rois+1)

    # Calculate x positions for each target within this group
    global aux_positions, aux_color
    space = 1/(len(targets)+1)
    group_positions = [aux_positions + i*space for i in range(len(targets))] # here we space them by 0.4 units.

    jitter = 0.02  # amount of horizontal jitter for individual scatter points

    # Boxplot for each ROI
    for i, roi in enumerate(roi_labels):
        ax.boxplot(all_roi_times[roi], positions=[group_positions[i]], widths=space, tick_labels=[f'{roi}'])

    # Scatter plot with jitter
    jitter = space*0.01 
    for i, roi in enumerate(roi_labels):
        ax.scatter(
            [group_positions[i] + np.random.uniform(-jitter, jitter) for _ in range(len(all_roi_times[roi]))],
            all_roi_times[roi],
            alpha=0.7,
            label=f'{group} {roi}'  # Avoid duplicate legend entries
        )
    
    ax.set_ylabel('Time Spent (s)')
    ax.set_title(f'Time spent in each area ({group} - {trial})')
    ax.legend(loc='best', fancybox=True, shadow=True, ncol=2)
    ax.grid(False)

    # Update the global positions variable
    aux_positions += 1

def count_alternations_and_entries(area_sequence):
    """
    Count the number of alternations and total area entries in a given sequence of visited areas.

    Args:
        area_sequence (list): Ordered list of visited areas.

    Returns:
        tuple: (Number of alternations, Total number of area entries)
    """
    # Remove consecutive duplicates (track only area **entrances**)
    filtered_seq = [area_sequence[i] for i in range(len(area_sequence)) if i == 0 or area_sequence[i] != area_sequence[i - 1]]
    
    # Remove 'other' from the sequence
    filtered_seq = [area for area in filtered_seq if area != "other"]

    total_entries = len(filtered_seq)  # Total number of area entrances
    alternations = 0

    for i in range(len(filtered_seq) - 2):
        if filtered_seq[i] != filtered_seq[i + 2] and filtered_seq[i] != filtered_seq[i + 1]:
            alternations += 1

    return alternations, total_entries

def plot_alternations(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None, outliers=[]) -> None:
    """
    Plot a boxplot of the proportion of alternations over total area entrances.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    folder = os.path.join(path, 'summary', group, trial)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    alternation_proportions = []

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        filename = os.path.basename(file_path)
        if any(outlier in filename for outlier in outliers):
            continue  # Skip files matching any outlier name
        df = pd.read_csv(file_path)

        if "location" not in df.columns:
            raise ValueError(f"File {file_path} does not contain a 'location' column.")

        area_sequence = df["location"].tolist()
        alternations, total_entries = count_alternations_and_entries(area_sequence)
        # print(f"Alternations: {alternations}, Total Entries: {total_entries}")

        if total_entries > 2:
            alternation_proportions.append(alternations / (total_entries-2)) # Exclude the first two entries
        else:
            alternation_proportions.append(0)  # Avoid division by zero

    # Calculate x positions for each target within this group
    global aux_positions, aux_color
    space = 1/(len(targets)+1)
    group_positions = [aux_positions + i*space for i in range(len(targets))] # here we space them by 0.4 units.
    color = colors[aux_color]

    jitter = 0.02  # amount of horizontal jitter for individual scatter points
    
    # Boxplot
    ax.boxplot(alternation_proportions, positions=[group_positions[0]], tick_labels=[f'{group}'])
    
    # Replace boxplots with scatter plots with jitter
    jitter = 0.05  # Adjust the jitter amount as needed
    ax.scatter([group_positions[0] + np.random.uniform(-jitter, jitter) for _ in range(len(alternation_proportions))], alternation_proportions, color=color, alpha=0.7,label="Alternation Proportion")

    ax.set_ylabel("Proportion of Alternations")
    ax.set_title(f"Proportion of Alternations ({group} - {trial})")

    ax.legend(loc="best", fancybox=True, shadow=True)
    ax.grid(False)

    # Update the global positions and color variables
    aux_positions += 1
    aux_color += len(targets)