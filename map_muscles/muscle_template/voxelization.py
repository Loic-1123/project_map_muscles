from _root_path import add_root 
add_root()

import numpy as np
import open3d as o3d

import map_muscles.muscle_template.xray_utils as xu

muscles = xu.get_femur_muscles(remove=True)
muscle = muscles[0]

