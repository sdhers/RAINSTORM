<div align="center">
  
# **RAINSTORM**
### Real & Artificial Intelligence Networks – Simple Tracker for Object Recognition Memory

![RAINSTORM Logo](docs/images/logo.png)

</div>

**RAINSTORM** is a tool for scoring object recognition memory in mice 🐭. It allows users to automate the analysis of recognition memory performance through training of artificial neural networks.

DeepLabCut analyzes video files and returns the position of the mouse's bodyparts... What we do next is up to us!

---
---

## **Features**  

- **Frame-by-Frame Behavioral Labeling**  
   Use this versatile tool for accurate manual scoring. On top of that, prepare your data to train an artificial neural network to recognize behaviors.  

- **Post-DeepLabCut Data Processing**  
   Clean up tracking glitches like disappearing body parts and ensure smooth, reliable data. 

- **Geometric Analysis for Exploration**  
   Leverage distance and angle metrics to identify exploration behavior with precision.  

- **Immobility Detection for Freezing Analysis**  
   Automatically label freezing behavior based on motion, a key indicator of memory performance.  

- **AI-Powered Automatic Labeling**  
   Train and utilize artificial neural networks to detect temporal sequences of exploration behavior.  

- **Visual Label Comparison**  
   Easily compare manual, geometric, and AI-generated labels using intuitive visualizations.

---
---

## **Installation**

1. **Install Miniconda (or Anaconda)**

Download and install Miniconda or Anaconda from the [official installation site](https://docs.anaconda.com/miniconda/install/).

2. **Install Visual Studio Code**

Download and install Visual Studio Code from the [official installation site](https://code.visualstudio.com/Download). During installation, ensure you add VS Code to your PATH (check the box if prompted).

3. **Clone the RAINSTORM Repository**

Open a terminal (e.g. Miniconda Prompt).

Navigate to your Desktop:

```bash
cd Desktop
```
  
Clone the repository (This step will create a folder named RAINSTORM on your Desktop):

```bash
git clone https://github.com/sdhers/rainstorm.git
```

4. **Set Up the Conda Environment**

Navigate to the rainstorm directory:

```bash
cd rainstorm
```
  
Create the Conda environment:
  
```bash
conda env create -f rainstorm_venv.yml
```

Activate the environment:

```bash
conda activate rainstorm
```

5. **Open VS Code**

Launch VS Code from the terminal:

```bash
code .
```
  
In VS Code, ensure the Python extension is installed:
  - Go to the Extensions view (```Ctrl+Shift+X``` or ```Cmd+Shift+X``` on macOS).
  - Search for "Python" and install the extension provided by Microsoft.

Open the ```0-First_steps.ipynb``` notebook.
  - When prompted to select a kernel, choose the ```rainstorm``` Conda environment among the ```Python Environments```.

6. **Start Exploring RAINSTORM**
  - Run the cells in 0-First_steps.ipynb to get started.

From now on, you can launch any of the notebooks from the RAINSTORM repository just by clicking on them in VS Code. The setup is complete!

---
---

## **Manual Labeling Tool**

Before getting to the automated part of the project, let me introduce you to the **RAINSTORM labeler** tool.

This simple python tool will let you label a video frame by frame, and get a precise register of what is happening (what behaviours are being displayed) on every moment of the recording.

As we already have our rainstorm environment created, all we need to do is open the terminal and:

1. **Activate the conda environment**

```bash
conda activate rainstorm
```

2. **Run the labeler**

```bash
python -m rainstorm.labeler
```

- A pop up window will appear, where we need to:

3. **Navigate to the video we want to label and select it**

- If you dont have a video available on your computer, you can find a demo inside the RAINSTORM repository on ```docs/examples/colabeled_video/Example_video.mp4```.
- You can also pick a labeled csv file if you already started labeling and want to pick up where you left off.

4. **Type the behaviors you would like to label**

- As we label the exploration of two objects, the presets are ```explore_left```, ```explore_right```, ```freezing``` and ```grooming```.

5. **Type the keyboard keys you'd like to use**

- One for each behaviour, the presets are ```4```, ```6```, ```f``` and ```g```.

6. **Start labeling**

- After a few seconds the labeler will open and the first frame will be displayed.
- Follow the instrucitons on the screen to navivate through the video and label each behavior as it happens.

Once you are done, exit and save the results, and a labeled csv file will be created on the selected video directory.

> [!TIP]
> The heavier the video, the longer it will take the program to process it. If it is too demanding for your computer, try compressing the video.

---
---

## **Pipeline**
The repository contains a series of Jupyter notebooks to go from raw pose estimation data to the visualization of your results:

```0-First_steps.ipynb```: Learn the basics and set up your data.

- This is the first notebook of the RAINSTORM project. Here you'll find the initial steps to prepare the data for analysis.
- If you dont have your pose estimation files yet (or you just want to try out the workflow), RAINSTORM comes with an example folder with pose estimation files from mice on a Novel Object Recognition (NOR) task.

---

```1-Prepare_positions.ipynb```: Process and clean bodypart position data.
- Filter out the frames where the mouse is not in the video.
- Points that have a low likelihood assigned by DLC are also filtered out, and data is interpolated and smoothed.
- Conveniently scale the video from pixels to cm.
- Return: We obtain .csv files with the correct, scaled positions of the mice.

![1-Prepare_positions](docs/images/1-Prepare_positions.png)

---

```2-Geometric_analysis.ipynb```: Perform geometric labeling of exploration and freezing.

- One way of finding out when the mouse is exploring an object is to use a geometric criteria:
  - If the mouse is close to the object (distance < 2.5 cm).
  - If the mouse is oriented towards the object (angle < 45°).

![2-Geometric_analysis](docs/images/2-Geometric_analysis.png)

---

```3-Create_Models.ipynb```: Train AI models for automatic behavioral labeling.

- Another way of finding out when the mouse is exploring is to train an artificial neural network with manually labeled data:

![3-Create_Models](docs/images/3-Create_models.png)

- Using TensorFlow, we were able to train models that are able to clasify a mouse's position into exploration.
- Among the models, we train a more complex LSTM network that is aware of frame sequences, and performs better as exploration is time dependant.
- The training learns from our own manual labeling, so it acquires the criteria of the users.

---

```4-Evaluate_Models.ipynb```: Assess and improve your trained models.

![4-Evaluate_Models](docs/images/4-Evaluate_models.png)

---

```5-Automatic_analysis.ipynb```: Automate labeling with your AI model.

![5-Automatic_analysis](docs/images/5-Automatic_analysis.png)

---

```6-Compare_Labels.ipynb```: Compare manual, geometric, and AI-generated labels.

- Once we have the manual, geometric and automatic labels, we can compare the performance of each on an example video
- Using a polar graph, we can see for each position the angle of approach and distance in which the mice is exploring the objects

![6-Compare_Labels](docs/images/6-Compare_labels.png)

---

```7-Seize_Labels.ipynb```: Extract and summarize your labeled data.

- We can use the best labels to evauate the memory performance of a single mouse:

![7-Seize_Labels_a](docs/images/7-Seize_labels_a.png)

- And finally, we can find differences in the exploration of objects for a group of trained mice (which was the obective all along):

![7-Seize_Labels_b](docs/images/7-Seize_labels_b.png)

---
---

## **Conclusions**
- This project, although already in use, is a work in progress that could significantly improve the way we analyze object exploration videos.
- If you wish to contact us, please do so: simple.torm@gmail.com
- © 2024. This project is openly licensed under the MIT License.

#### Thanks for exploring us!

![mouse_exploring](docs/images/mouse_exploring.gif)
