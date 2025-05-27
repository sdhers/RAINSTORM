# ROI Selector Application

This application allows you to select, define, and save Regions of Interest (ROIs) and points on a merged reference image generated from one or more video files. It supports drawing rectangles, circles, points, and defining a real-world scale.

## Features

* **Video Frame Merging**: Automatically creates a single averaged reference image from the first, middle, and last frames of a single video, or the first frame of multiple videos.
* **Interactive ROI Drawing**:
    * Draw **Rectangles**: Define custom rectangular ROIs with adjustable size and rotation.
    * Draw **Circles**: Define circular ROIs with adjustable radius.
    * Mark **Points**: Pinpoint specific locations.
* **Dynamic Previews**: See your ROI as you draw, move, or resize it.
* **Intuitive Controls**:
    * **Drawing**: Left-click and drag for rectangles, Shift + Left-click and drag for circles, single Left-click for points.
    * **Modification**: Right-click and drag to move, scroll wheel to resize, Ctrl + scroll wheel to rotate.
    * **Zoom**: Shift + scroll wheel to zoom in/out on the cursor.
    * **Nudging**: WASD keys to fine-tune the position of the active ROI or cursor.
* **Scale Definition**: Draw a line and enter its real-world length to establish a pixels-per-unit scale for your image.
* **Save/Load ROIs**: Save all defined ROIs and scale to a JSON file and load them for later editing or use.
* **Undo/Clear**: Option to undo the last drawn ROI or clear all ROIs.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ROI_Selector.git](https://github.com/yourusername/ROI_Selector.git)
    cd ROI_Selector
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install opencv-python numpy
    ```

## Usage

1.  **Run the application:**
    ```bash
    python main.py
    ```
2.  **Select Video Files**: A file dialog will appear. Select one or more video files (`.mp4`, `.avi`, `.mkv`, `.mov`) that you want to use as a reference for drawing ROIs.
3.  **Load Existing ROIs (Optional)**: After selecting videos, you'll be asked if you want to load a previously saved `ROIs.json` file. If you do, a file dialog will open to select your JSON file.
4.  **Instructions Window**: An instruction window will pop up detailing all controls. Read it carefully!

### Controls:

* **Drawing a Rectangle**:
    * **Left-click and drag**: Draw a rectangle.
    * **Hold `Ctrl` while dragging**: Enforce a square shape.
* **Drawing a Circle**:
    * **Hold `Shift` + Left-click and drag**: Draw a circle. The initial click is the center, and dragging defines the radius.
* **Marking a Point**:
    * **Single Left-click**: Mark a single point.
* **Drawing a Scale Line**:
    * **Hold `Alt` + Left-click and drag**: Draw a line to define a real-world scale. After drawing, press `Enter` to input the real length.
* **Confirming an ROI/Point/Scale**:
    * Press `Enter` (⏎): Saves the actively drawn rectangle, circle, point, or confirms the scale line. You will be prompted to name the ROI/point or enter the real length for scale.
* **Moving an Active ROI (before confirming)**:
    * **Right-click and drag**: Move the active rectangle or circle.
* **Resizing an Active ROI (before confirming)**:
    * **Scroll wheel**: Resize the active rectangle (adjusts width/height) or circle (adjusts radius).
* **Rotating an Active Rectangle ROI (before confirming)**:
    * **`Ctrl` + Scroll wheel**: Rotate the active rectangle.
* **Zooming the View**:
    * **`Shift` + Scroll wheel**: Zoom in/out on the cursor position.
* **Nudging (Fine Adjustment)**:
    * Press `W`, `A`, `S`, `D` keys: Nudge the active ROI (rectangle/circle center) or the cursor (if no active ROI) by one pixel.
* **Undo Last Action**:
    * Press `B` key: Erase the last saved ROI (circle, then rectangle, then point).
* **Erase All ROIs**:
    * Press `E` key: Clears all saved ROIs, points, and circles. (Requires confirmation).
* **Quit Application**:
    * Press `Q` key: Exit the application. You will be prompted to save your current ROIs.

## Output

Upon quitting and choosing to save, a `ROIs.json` file will be created in the directory of your first selected video file (or in a `data/` folder if no videos were selected). This JSON file contains:

* `frame_shape`: The width and height of the reference frame used for ROI definition.
* `scale`: The defined pixels-per-unit scale (if set).
* `areas`: A list of defined rectangular ROIs, each with `name`, `type`, `center`, `width`, `height`, and `angle`.
* `points`: A list of defined points, each with `name`, `type`, and `center`.
* `circles`: A list of defined circular ROIs, each with `name`, `type`, `center`, and `radius`.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or bug fixes.

## License

This project is open-source. (Consider adding a specific license, e.g., MIT, Apache 2.0)