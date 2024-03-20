from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.projection_on_plane as pp

femur_muscles = xu.get_femur_muscles()
    
layers_names = set([muscle['layer_name'].iloc[0] for muscle in femur_muscles])
"""
{'LH FeTi anterior acc flexor', 'LH FeCl ltm2', 
'LH FeTi extensor', 'LH FeTi posterior acc flexor', 
'LH FeTi flexor', 'LH TrFe'}
"""

# take one muscle

trfe = femur_muscles[-1]
trfe = xu.add_lines_to_df(trfe)
trfe.describe()
type(trfe['line'].iloc[0])
lines = trfe['line']
type(lines)
lines.describe()
line = lines.iloc[0]
type(line)

colors = vf.get_random_color_map(femur_muscles)
vf.plot_fibers([trfe], colors)

v1 = np.array([1.,1.,1.])
v2 = np.array([1.,.0,.0])

u1, u2 = pp.orthonormal_vectors(v1, v2)

projection = pp.projected_segments(trfe['line'], [u1, u2])

plt.show()
