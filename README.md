<div align="center">
  
# **RAINSTORM**
### Real & Artificial Intelligence Networks – Simple Tracker for Object Recognition Memory

![RAINSTORM Logo](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/logo.png)

</div>

**RAINSTORM** is a tool for scoring object recognition memory in mice 🐭. It allows users to automate the analysis of recognition memory performance through training of artificial neural networks.

---

## **Features**  

- **Frame-by-Frame Behavioral Labeling**  
   Use this versatile tool for manual scoring or to train an artificial neural network to recognize behaviors.  

- **Post-DeepLabCut Data Processing**  
   Clean up tracking glitches like disappearing body parts and ensure smooth, reliable data.  

- **Geometric Analysis for Exploration**  
   Leverage distance and angle metrics to identify exploration behavior with precision.  

- **Immobility Detection for Freezing Analysis**  
   Automatically label freezing behavior based on motionlessness, a key indicator of memory performance.  

- **AI-Powered Automatic Labeling**  
   Train and utilize artificial neural networks to detect temporal sequences of exploration behavior.  

- **Visual Label Comparison**  
   Easily compare manual, geometric, and AI-generated labels using intuitive visualizations.  

---

### Future steps

- Multianimal labeling for social memories
- Apply detection of moving objects for dinamic maze designs

## **Installation**  

1. **Install Miniconda or Anaconda**  
   Download Miniconda or Anaconda from the [official installation guide](https://docs.anaconda.com/miniconda/install/).  

2. **Clone the Repository**  
   Download or clone the **RAINSTORM** repository to your local machine:  
   ```bash
   git clone https://github.com/sdhers/RAINSTORM.git
   cd rainstorm

3. **Set Up the Conda Environment**  
   Create a dedicated Conda environment for RAINSTORM:  
   ```bash
   conda env create -f rainstorm_venv.yml

4. **Activate the Environment**  
   Activate the newly created environment to start using RAINSTORM:
   ```bash
   conda activate rainstorm
5. **Run the Jupyter Notebooks**
  Launch Jupyter Notebook and start exploring the project’s capabilities.

## **Pipeline**
The repository contains a series of Jupyter notebooks that guide you through the pipeline:
```0-First_steps.ipynb```: Learn the basics and set up your data.
```1-Prepare_positions.ipynb```: Process and clean body-part position data.
```2-Geometric_analysis.ipynb```: Perform geometric labeling of exploration and freezing.
```3-Create_Models.ipynb```: Train AI models for automatic behavioral labeling.
```4-Evaluate_Models.ipynb```: Assess and improve your trained models.
```5-Automatic_Labeling.ipynb```: Automate labeling with your AI model.
```6-Compare_Labels.ipynb```: Compare manual, geometric, and AI-generated labels.
```7-Seize_Labels.ipynb```: Extract and summarize your labeled data.

---
---
---
older readme

- DeepLabCut analyzes video files and returns a .H5 file with the position of the mouse's bodyparts (along with two objects, in the case of object exploration). What we do next is up to us!

## Manage_H5

- It is important to filter from the position file the frames where the mouse is not in the video
- Points that have a low likelihood assigned by DLC are filtered and data is smoothed
- Also, it is convenient to scale the video from pixels to cm
- Return: We obtain .csv files with the correct, scaled positions of the mice

![Example Manage_H5](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/1-Manage_H5.png)

## Geometric_Labeling

- One way of finding out when the mouse is exploring an object is to use a geometric criteria:
  - If the mouse is close to the object (distance < 2.5 cm)
  - If the mouse is oriented towards the object (angle < 45°)

![Example Geolabels](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/2-Geometric_Labeling.png)

## Automatic_Labeling

- Another way of finding out when the mouse is exploring is to train an artificial neural network with manually labeled data:

![Example Autolabels](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/3a-Create_Models.png)

Using TensorFlow, we were able to train models that are able to clasify a mouse's position into exploration

![Example Autolabels_2](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/3a-Create_Models_simple.png)

Among the models, we trained a more complex LSTM network that is aware of frame sequences, and performs better as exploration is time dependant.
It trains based on our own manual labeling, so it acquires the criteria of the users.

## Compare_Labels

- Once we have the manual, geometric and automatic labels, we can compare the performance of each on an example video:

![Example compare_1](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/4-Compare_Labels_line.png)

Using a polar graph, we can see for each position the angle of approach and distance in which the mice is exploring the objects
- For a single video:

![Example compare_1](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/4-Compare_Labels_polar.png)

- Or for many videos together:

![Example compare_2](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/4-Compare_Labels_polar_all.png)

#### Since the automatic method learns to detect exploration unrestricted by the angle and distance to the object, it tends to be more accurate (Although, let's be honest... I chose to show you the best numbers I've ever gotten).

![Example compare_3](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/4-Compare_Labels_result.png)

## Seize_Labels

- We can use the best labels to evauate the performance of a mouse during the different sessions:

![Example seize_1](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/5-Seize_Labels_example.png)

- And finally, we can find differences in the exploration of objects for a group of trained mice (which was the obective all along):

![Example seize_2](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/5-Seize_Labels_experiment.png)

# In conclusion
- This project, although already in use, is a work in progress that could significantly improve the way we analyze object exploration videos.
- If you wish to contact us, please do so: dhers.santiago@gmail.com
- © 2024. This project is openly licensed under the MIT License.

#### Thanks for exploring us!

![Final_gif](https://github.com/sdhers/RAINSTORM/blob/main/docs/images/mouse_exploring.gif)
