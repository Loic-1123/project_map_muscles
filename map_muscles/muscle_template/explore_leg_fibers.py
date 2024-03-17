from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import pandas as pd
pd.options.mode.copy_on_write = True

import map_muscles.muscle_template.xray_utils as xu
import matplotlib.pyplot as plt
import numpy as np


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
# extract the thco layers names
thco_ids = [layer for layer in layers_names if layer.startswith('LH ThCo')]
thcos = [lh[lh['layer_name'] == thco] for thco in thco_ids]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# for each thco, choose a color from a color palette
colors = plt.cm.viridis(np.linspace(0, 1, len(thcos)))


for thco, c in zip(thcos, colors):
       thco.loc[:,'line'] = thco.apply(lambda row: xu.create_line(row['pointA'], row['pointB']), axis=1)
       lines = thco['line']
       # use 1 same color for each thco
       for line, color in zip(lines, c):
              x, y, z = line
              ax.plot(x, y, z, color=c)

       ax.set_xlabel('X Label')
       ax.set_ylabel('Y Label')
       ax.set_zlabel('Z Label')

       # show a label for each thco
       ax.text(x[-1], y[-1], z[-1], thco['layer_name'].values[0], color=c)

       plt.legend()

plt.show()
