from _root_path import add_root
add_root()

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


vec = np.array([1,1,1])

# compute pitch and yaw
pitch = np.arcsin(vec[2]/norm(vec))

yaw = np.arctan2(vec[1], vec[0])

pitch_deg = np.degrees(pitch)
yaw_deg = np.degrees(yaw)

print(f"pitch: {pitch}, yaw: {yaw}")
print(f"pitch_deg: {pitch_deg}, yaw_deg: {yaw_deg}")




