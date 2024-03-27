# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:28:57 2024

@author: dhers
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class Point:
    def __init__(self, df, table):
        x = df[f'{table}_x']
        y = df[f'{table}_y']
        self.positions = np.dstack((x, y))[0]

def plot_position_video(file, save_path, maxDistance=2.5, max_nose_points=10, start_seconds = 20, end_seconds = 25, fps = 25):
    
    df = pd.read_csv(file)
    df = df.loc[start_seconds*fps:end_seconds*fps]
    
    filename = os.path.splitext(os.path.basename(file))[0]

    obj1 = Point(df, 'obj_1')
    obj2 = Point(df, 'obj_2')
    nose = Point(df, 'nose')
    head = Point(df, 'head')
    neck = Point(df, 'neck')
    L_ear = Point(df, 'L_ear')
    R_ear = Point(df, 'R_ear')
    body = Point(df, 'body')
    tail_1 = Point(df, 'tail_1')
    tail_2 = Point(df, 'tail_2')
    tail_3 = Point(df, 'tail_3')
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Horizontal position (cm)")
    ax.set_ylabel("Vertical position (cm)")
    plt.title(f"Analysis of {filename}")
    plt.tight_layout()
    ax.axis('equal')
    
    ax.set_xlim(0, 35)  # Set x-axis limits
    ax.set_ylim(0, 21)  # Set y-axis limits

    ax.plot(*obj1.positions[0], "s", lw = 20, label = "Object 1", color = "blue", markersize = 22, markeredgecolor = "darkblue")
    ax.plot(*obj2.positions[0], "o", lw = 20, label = "Object 2", color = "red", markersize = 25, markeredgecolor = "darkred")

    ax.add_artist(Circle(obj1.positions[0], 2.5, color = "orange", alpha = 0.3))
    ax.add_artist(Circle(obj2.positions[0], 2.5, color = "orange", alpha = 0.3))
    # ax.legend(bbox_to_anchor=(0, 0, 1, 1), ncol=2, loc='upper left', fancybox=True, shadow=True)

    def update(frame):
        ax.clear()  # Clear the previous plot
        
        ax.set_xlabel("Horizontal position (cm)")
        ax.set_ylabel("Vertical position (cm)")
        plt.title("Dynamic exploration of two objects")
        plt.tight_layout()
        ax.axis('equal')
        
        ax.set_xlim(0, 35)  # Set x-axis limits
        ax.set_ylim(0, 21)  # Set y-axis limits
    
        ax.plot(*obj1.positions[0], "s", lw = 20, label = "Object 1", color = "blue", markersize = 22, markeredgecolor = "darkblue")
        ax.plot(*obj2.positions[0], "o", lw = 20, label = "Object 2", color = "red", markersize = 25, markeredgecolor = "darkred")
    
        ax.add_artist(Circle(obj1.positions[0], 2.5, color = "orange", alpha = 0.3))
        ax.add_artist(Circle(obj2.positions[0], 2.5, color = "orange", alpha = 0.3))
        # ax.legend(bbox_to_anchor=(0, 0, 1, 1), ncol=2, loc='upper left', fancybox=True, shadow=True)
        
        start_frame = max(0, frame - max_nose_points)  # Start frame for plotting nose points
        ax.plot(*nose.positions[start_frame:frame].T, "o", color="green", alpha=0.5)
        ax.plot(*nose.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*head.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*neck.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*L_ear.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*R_ear.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*body.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*tail_1.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*tail_2.positions[frame].T, ".", color="grey", alpha=0.75)
        ax.plot(*tail_3.positions[frame].T, ".", color="grey", alpha=0.75)
        
        # Add lines between body parts (for example, from nose to head)
        ax.plot([nose.positions[frame][0], head.positions[frame][0]],
                [nose.positions[frame][1], head.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([nose.positions[frame][0], L_ear.positions[frame][0]],
                [nose.positions[frame][1], L_ear.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([nose.positions[frame][0], R_ear.positions[frame][0]],
                [nose.positions[frame][1], R_ear.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([head.positions[frame][0], neck.positions[frame][0]],
                [head.positions[frame][1], neck.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([neck.positions[frame][0], body.positions[frame][0]],
                [neck.positions[frame][1], body.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([body.positions[frame][0], tail_1.positions[frame][0]],
                [body.positions[frame][1], tail_1.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([tail_1.positions[frame][0], tail_2.positions[frame][0]],
                [tail_1.positions[frame][1], tail_2.positions[frame][1]], '-', color='black', alpha=0.5)
        ax.plot([tail_2.positions[frame][0], tail_3.positions[frame][0]],
                [tail_2.positions[frame][1], tail_3.positions[frame][1]], '-', color='black', alpha=0.5)

    ani = FuncAnimation(fig, update, frames=len(nose.positions), interval=14.5, repeat=True)
    
    ani.save(save_path, writer='imagemagick')  # Requires ImageMagick to be installed
    
    #plt.show()
    
    return ani

example_path = r'C:/Users/dhers/Desktop/Videos_NOR/2022-02_TORM_3h/TS/position/2022-02_TORM_3h_TS_C3_B_L_position.csv'
output_path = r'C:/Users/dhers/Desktop/mouse_exploring.gif'

# Example usage
plot_position_video(example_path, output_path, maxDistance=2.5, max_nose_points=10, start_seconds = 22.9, end_seconds = 33.6, fps = 25)





