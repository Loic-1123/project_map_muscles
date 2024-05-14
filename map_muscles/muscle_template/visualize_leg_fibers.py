from _root_path import add_root
add_root()

import pandas as pd
pd.options.mode.copy_on_write = True

import map_muscles.muscle_template.xray_utils as xu
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

"""
This script visualizes the fibers of leg muscles.
Two visualization are shown:
1. All fibers of the leg muscles
2. The fibers related to the femur muscles
"""

def plot_lines(ax, lines, c):
       """
       Plot lines in a 3D plot.

       Parameters:
       - ax: The 3D axes object to plot on.
       - lines: A list of lines, where each line is a pair of tuple of x, y, and z coordinates, shape(n, 2, 3) # n lines, 2 points per line, 3 coordinates per point.
       - c: The color of the lines.

       Returns:
       None
       """
       for line in lines:
              x, y, z = line
              ax.plot(x, y, z, color=c)

def plot_text_label(ax, lines, label, c):
       x, y, z = lines.iloc[-1]
       ax.text(x[-1], y[-1], z[-1], label, color=c)

def set_xyz_labels(ax):
       ax.set_xlabel('X')
       ax.set_ylabel('Y')
       ax.set_zlabel('Z')

def get_shuffled_color_ids(parts):
       """
       Generate shuffled color IDs for each part.

       Parameters:
       parts (list): A list of parts.

       Returns:
       numpy.ndarray: An array of shuffled color IDs.
       """
       color_ids = np.linspace(0, 1, len(parts))
       np.random.shuffle(color_ids)
       return color_ids

def get_random_color_map(parts, rgb_only=False, palette=plt.cm.hsv):
       """
       Generate a random color map for the given parts.

       Parameters:
       parts (list): A list of parts.
       rgb_only (bool, optional): If True, return only RGB values. Defaults to False.
       palette (matplotlib colormap, optional): The colormap to use for generating colors. Defaults to plt.cm.hsv.

       Returns:
       numpy.ndarray: An array of colors representing the color map.
       """
       color_ids = get_shuffled_color_ids(parts)
       colors = palette(color_ids)

       if rgb_only:
              colors = colors[:, :-1]

       return colors

def get_linear_color_map(parts, palette=plt.cm.hsv):
       """
       Generate a linear color map based on the number of parts.

       Parameters:
       parts (int): The number of parts to generate colors for.
       palette (matplotlib.colors.Colormap, optional): The colormap to use for generating colors. Defaults to plt.cm.hsv.

       Returns:
       numpy.ndarray: An array of colors corresponding to each part.
       """
       color_ids = np.linspace(0, 1, len(parts))
       return palette(color_ids)

def plot_fibers(parts, colors, plot_labels=False):
       """
       Plot fibers in a 3D space.

       Parameters:
       - parts (list): List of dataframes representing different parts of the fibers.
       - colors (list): List of colors corresponding to each part.
       - plot_labels (bool): Flag indicating whether to plot labels for each part.

       Returns:
       None
       """

       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       set_xyz_labels(ax)

       for part, c in zip(parts, colors):
              part.loc[:,'line'] = part.apply(lambda row: xu.create_line(row['pointA'], row['pointB']), axis=1)
              lines = part['line']
              plot_lines(ax, lines, c)

              if plot_labels:
                     plot_text_label(ax, lines, part['layer_name'].iloc[0], c)

if __name__ == "__main__":
       lh = xu.get_leg(leg='LH')

       layers_names = lh['layer_name'].unique()
       """
       array(['LH ThCo 1', 'LH ThCo 2', 'LH ThCo 3', 'LH ThCo? 4',
              'LH tergotrochanter', 'LH ThTr 2', 'LH ThTr 3 (same as 2?)',
              'LH ThTr 4', 'LH ThTr 5 (same as 4?)', 'LH CoTr 1', 'LH CoTr 2',
              'LH CoTr 3', 'LH CoTr 4', 'LH CoTr 5', 'LH TrFe', 'LH FeTi flexor',
              'LH FeTi anterior acc flexor', 'LH FeTi posterior acc flexor',
              'LH FeTi extensor', 'LH FeCl ltm2', 'LH TiCl ltm1',
              'LH TiTa 1 (detached origin)', 'LH TiTa 2',
              'LH TiTa 3 (detached origin)'], dtype=object)
       """

       part_ids = [layer for layer in layers_names]

       leg_parts = [lh[lh['layer_name'] == part] for part in part_ids]

       colors = get_random_color_map(leg_parts, palette=plt.cm.viridis)

       plot_fibers(leg_parts, colors)

       fe_ids = [layer for layer in layers_names if 'Fe' in layer]       
       """
       ['LH TrFe', # removed
       'LH FeTi flexor', 
       'LH FeTi anterior acc flexor', 
       'LH FeTi posterior acc flexor', 
       'LH FeTi extensor', 
       'LH FeCl ltm2']
       """

       femur_fibers = xu.get_femur_muscles(remove=True)

       colors = get_random_color_map(femur_fibers)

       plot_fibers(femur_fibers, colors, plot_labels=True)

       plt.show()