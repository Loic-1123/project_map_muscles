from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import pandas as pd
pd.options.mode.copy_on_write = True

import map_muscles.muscle_template.xray_utils as xu
import matplotlib.pyplot as plt
import numpy as np



def plot_lines(ax, lines, c):
       for line in lines:
              x, y, z = line
              ax.plot(x, y, z, color=c)

def plot_text_label(ax, lines, label, c):
       x, y, z = lines.iloc[-1]
       ax.text(x[-1], y[-1], z[-1], label, color=c)

def set_xyz_labels(ax):
       ax.set_xlabel('X Label')
       ax.set_ylabel('Y Label')
       ax.set_zlabel('Z Label')


def plot_fibers(parts, colors):
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')

       for part, c in zip(parts, colors):
              part.loc[:,'line'] = part.apply(lambda row: xu.create_line(row['pointA'], row['pointB']), axis=1)
              lines = part['line']
              plot_lines(ax, lines, c)

              set_xyz_labels(ax)

              plot_text_label(ax, lines, part['layer_name'].iloc[0], c)


def get_shuffled_color_ids(parts):
       color_ids = np.linspace(0, 1, len(parts))
       np.random.shuffle(color_ids)
       return color_ids

def get_random_color_map(parts, palette=plt.cm.hsv):
       color_ids = get_shuffled_color_ids(parts)
       return palette(color_ids)

def get_linear_color_map(parts, palette=plt.cm.hsv):
       color_ids = np.linspace(0, 1, len(parts))
       return palette(color_ids)

if __name__ == "__main__":
       lh = xu.get_leg(leg='LH')
       lh.describe()

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

       plt.show()

       # isolate fibers related to femur: layer names containing 'Fe'
       fe_ids = [layer for layer in layers_names if 'Fe' in layer]
       femur_fibers = xu.get_femur_muscles()

       colors = get_random_color_map(femur_fibers)

       plot_fibers(femur_fibers, colors)

       plt.show()

       # nb muscles related to femur
       len(femur_fibers)
       # 6

