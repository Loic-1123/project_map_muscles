from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import pandas as pd
import map_muscles.muscle_template.xray_utils as xu

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

