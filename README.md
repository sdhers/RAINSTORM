-----
# **RAINSTORM**

### Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory

**A complete toolkit for analyzing rodent exploratory behavior in object recognition tasks.**

![RAINSTORM Logo](examples/images/logo.png)

**RAINSTORM** is a Python-based tool for scoring exploratory behavior in rodents 🐭. It takes pose-estimation data (e.g., from DeepLabCut) and provides a full workflow to process, analyze, and visualize recognition memory performance, from manual labeling to AI-powered automation.

-----

## 📋 Table of Contents

  * [Features](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#features)
  * [Installation](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#installation)
  * [Usage](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#usage)
      * [Video Handling](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#video-handling)
      * [Rainstorm Behavioral Labeler](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#rainstorm-behavioral-labeler)
      * [The Pipeline](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#the-rainstorm-pipeline)

  * [Contributing](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#contributing)
  * [License](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#license)
  * [Contact](https://github.com/sdhers/RAINSTORM?tab=readme-ov-file#contact)

-----

## Features

  * **🎯 Frame-by-Frame Behavioral Labeling:** A versatile tool for precise manual scoring and for generating training data for your AI models.
  * **🔧 Pre & Post-DLC Data Processing:** Align video points, clean tracking glitches, and interpolate data for smooth and reliable analysis.
  * **📐 Geometric Analysis:** Automatically identify object exploration using distance and angle metrics.
  * **🧊 Immobility Detection:** Label freezing behavior based on motion, a key indicator in memory studies.
  * **🤖 AI-Powered Automatic Labeling:** Train and deploy neural networks (including LSTMs) to automatically detect complex exploration patterns.
  * **📊 Visual Label Comparison:** Easily compare manual, geometric, and AI-generated labels with intuitive visualizations.

-----

## Installation

### Prerequisites

First, ensure you have the following software installed on your system.

  * [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) (or Anaconda)
  * [Visual Studio Code](https://code.visualstudio.com/Download)
  * [Git](https://git-scm.com/downloads)

> [\!TIP]
> During the Miniconda installation, it is recommended to select the option to **add Conda to your system's PATH**. This will make it easier to run `conda` commands from any terminal.
> Make sure to reboot your computer to ensure the new software is properly installed.

### Setup Steps

1.  **Clone the Repository**
    Open a terminal (or Miniconda Prompt) and run the following command.
    ```bash
    git clone https://github.com/sdhers/rainstorm.git
    ```
    This will create a `rainstorm` folder in your current directory.

2.  **Set Up the Conda Environment**
    Navigate into the cloned directory and create the dedicated environment from the provided file:
    ```bash
    cd rainstorm
    conda env create -f rainstorm_venv.yml
    ```

Once the environment is ready, you can activate it by running ```conda activate rainstorm```.

3.  **Launch VS Code & Select Kernel**
    Launch VS Code from the terminal:

    ```bash
    code .
    ```
    
    In VS Code, ensure the Python extension is installed:
    - Go to the Extensions view (```Ctrl+Shift+X``` or ```Cmd+Shift+X``` on macOS).
    - Search for "Python" and install the extension provided by Microsoft.
  
    Open the ```0-Video_handling.ipynb``` notebook.
    - When prompted to select a kernel, choose the ```rainstorm``` Conda environment among the ```Python Environments```.

You are all set\! You can now run the notebooks to explore the RAINSTORM workflow.

-----

## 💻 Usage

RAINSTORM offers two main functionalities: a full analysis pipeline using Jupyter notebooks and a standalone tool for manual video labeling.

-----

### ✍️ RAINSTORM Behavioral Labeler

For precise, frame-by-frame annotation, use the **RAINSTORM Behavioral Labeler**.

2.  **Run the file `1-Behavioral_labeler.ipynb`**

3.  **Select the video you want to label**
  
5.  **(Optional) Load a previous labeling csv file**
      * If you want to pick up where you left off labeling the video.

6.  **Select (confirm) the behaviors you want to label, and their keys.**

      * Enter the behaviors you want to score (e.g., `exp_1, exp_2, freezing, grooming`).

> [\!WARNING]
> Keys should be unique, single characters, different from the operant and fixed control keys: (Quit: 'q', Zoom In: '+', Zoom Out: '-', Margin Toggle: 'm')

7.  **Start Labeling\!**
    The video will load, and you can begin annotating frame by frame using the keys you defined.

-----

### 🔬 The RAINSTORM Pipeline

The core of this project is a series of Jupyter notebooks designed to guide you from raw data to final results.

-----

#### `2a-Prepare_positions.ipynb`

🧹 **Process and clean bodypart position data.**

  * Filters out frames with low tracking likelihood from DeepLabCut.
  * Interpolates and smooths data to correct glitches.
  * Scales coordinates from pixels to a more convenient unit (e.g., cm).
  * **Output:** Clean, scaled `.csv` files ready for analysis.

![1-Prepare_positions](examples/images/1-Prepare_positions.png)

-----

#### `2b-Geometric_analysis.ipynb`

📐 **Perform geometric labeling of exploration and freezing.**

  * Applies a simple geometric rule for exploration:
      * Distance to object \< `2.5 cm`
      * Angle towards object \< `45°`
  * Identifies freezing behavior based on lack of movement.

![2-Geometric_analysis](examples/images/2-Geometric_analysis.png)

-----

#### `3a-Create_Models.ipynb`

🤖 **Train AI models for automatic behavioral labeling.**

  * Uses your manually labeled data to train TensorFlow models.
  * Includes an LSTM network that considers temporal sequences for higher accuracy.
  * Evaluates model performance against human labelers using Principal Components Analysis (PCA).

![3-Create_Models](examples/images/3-Create_models.png)

![4-Evaluate_models_a](examples/images/4-Evaluate_models_a.png)

![4-Evaluate_models_b](examples/images/4-Evaluate_models_b.png)

-----

#### `3b-Automatic_analysis.ipynb`

🧠 **Automate labeling with your trained AI model.**

  * Applies your best-performing model to label large datasets.
  * Generates comparative visualizations (like polar graphs) to contrast manual, geometric, and AI-driven labels.

![6-Compare_Labels](examples/images/6-Compare_labels.png)

-----

#### `4-Seize_Labels.ipynb`

📊 **Extract, summarize, and visualize your final data.**

  * Calculates key metrics like the Discrimination Index.
  * Generates publication-ready plots to compare behavior across different experimental groups and sessions (Habituation, Training, Test).

![7-Seize_Labels_ts](examples/images/7-Seize_labels_ts.png)

![7-Seize_Labels_tr](examples/images/7-Seize_labels_tr.png)

![7-Seize_Labels_hab](examples/images/7-Seize_labels_hab.png)

-----

## 🤝 Contributing

This project is a work in progress. Contributions are welcome\! If you have suggestions or find a bug, please feel free to open an issue or submit a pull request.

-----

## 📜 License

This project is openly licensed under the **MIT License**. See the `LICENSE` file for more details.
© 2024, sdhers.

-----

## 📫 Contact

For any questions or collaborations, please reach out to sdhers@fbmc.fcen.uba.ar.

*Thanks for exploring RAINSTORM\!*

![mouse_exploring](examples/images/mouse_exploring.gif)

\</div\>

-----
