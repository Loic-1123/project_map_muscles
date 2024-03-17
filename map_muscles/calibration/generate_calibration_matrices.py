import numpy as np
import cv2
from cv2 import aruco
import pickle
from pathlib import Path

import json

abs_path = Path(__file__).parent
# save the points to pkl
calib_pts_dir = abs_path / "data/calib_points"

with open(calib_pts_dir/"muscle_pts.pkl", "rb") as f:
    muscle_calib_pts = pickle.load(f)
with open(calib_pts_dir/"kinematics_pts.pkl", "rb") as f:
    kin_calib_pts = pickle.load(f)

# Get te essential matrix
pts_muscle = np.array(muscle_calib_pts).astype(np.int64).reshape(-1, 2)
pts_kin = np.array(kin_calib_pts).astype(np.int64).reshape(-1, 2)
F_comp, mask = cv2.findFundamentalMat(pts_muscle, pts_kin, cv2.FM_LMEDS)

F = np.array([[ 1.58392429e-06,  2.42491890e-08, -7.40022349e-04],
       [-2.05925281e-08,  1.54211788e-06, -4.30077932e-04],
       [-1.52129280e-03, -1.03792264e-03,  1.00000000e+00]])

#print(F-F_comp)
#F = F_comp
#get epioles
U, S, V = np.linalg.svd(F)
e_muscle = V[-1, :]
e_muscle = e_muscle/e_muscle[-1]

e_kin = U[:, -1]
e_kin = e_kin/e_kin[-1]

dists_muscle = np.abs(np.linalg.norm(pts_muscle-e_muscle[:2], axis=-1))
dists_kin= np.abs(np.linalg.norm(pts_kin-e_kin[:2], axis=-1))

muscle_kin_dist_ratios = dists_muscle/dists_kin
muscle_kin_dist_ratio = np.median(muscle_kin_dist_ratios)

# get four reference points
# four extreme points in the kinematic image
min_y_pts = np.argmin(pts_kin[:, 1])
max_y_pts = np.argmax(pts_kin[:, 1])
min_x_pts = np.argmin(pts_kin[:, 0])
max_x_pts = np.argmax(pts_kin[:, 0])

ref_kin_pts = pts_kin[[min_y_pts, max_y_pts, min_x_pts, max_x_pts]]
ref_muscle_pts = pts_muscle[[min_y_pts, max_y_pts, min_x_pts, max_x_pts]]

calib_dict = {
    "F": F.tolist(),
    "e_muscle": e_muscle.tolist(), "e_kin": e_kin.tolist(),
    "muscle_kin_ratio": muscle_kin_dist_ratio.tolist(),
    "ref_kin_pts": ref_kin_pts.tolist(), "ref_muscle_pts": ref_muscle_pts.tolist()
              }

out_calib_path = abs_path/"calibration_parameters.json"
with open(out_calib_path, "w") as f:
    json.dump(calib_dict, f)
