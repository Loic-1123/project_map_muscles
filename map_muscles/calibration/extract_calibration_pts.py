from cv2 import aruco
import cv2
import numpy as np
from map_muscles.calibration.config.aruco_board import CHARUCO_BOARD
from map_muscles.calibration.calibration_utils import (compare_board_image,
                                                       yield_images_paths)
import yaml
import pickle
from pathlib import Path
abs_path = Path(__file__).parent

# Load the camera calibration paths
with open(abs_path / "config/calib_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

if config["save_images"]:
    img_out_path = abs_path / "data/calib_images"
    img_out_path.mkdir(exist_ok=True)

nas_folder = Path(config["nas_base_folder_path"])
full_muscle_image_paths = [nas_folder / path
                           for path in config["muscle_images_paths"]]
full_kinematic_image_paths = [nas_folder / path
                              for path in config["kinematic_images_paths"]]

charuco_detector = aruco.CharucoDetector(CHARUCO_BOARD)

kinematic_camera_imgpts = []
muscle_camera_imgpts = []
selected_calib_img_id = 0

for m_path, k_path, k_channel in yield_images_paths(
        full_muscle_image_paths,
        full_kinematic_image_paths,
        config["kin_to_muscle_div_factor"],
        2000):

    m_frame = cv2.imread(str(m_path))
    m_frame = cv2.flip(m_frame, 1)
    #m_frame = cv2.normalize(m_frame, None)
    m_frame = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)
    m_frame = np.clip(m_frame * 50.0, 0, 255).astype(np.uint8)
    k_frame = cv2.imread(str(k_path))[:, :, k_channel]

    (muscle_charuco_corners, muscle_charuco_ids, _, _) = \
        charuco_detector.detectBoard(m_frame)

    if np.any(muscle_charuco_ids):
        (kinematic_charuco_corners, kinematic_charuco_ids, _, _) = \
            charuco_detector.detectBoard(k_frame)
        if np.any(kinematic_charuco_ids):
            overlapping_ids = np.intersect1d(kinematic_charuco_ids,
                                             muscle_charuco_ids)
            for i in overlapping_ids:
                muscle_camera_imgpts.append(muscle_charuco_corners[
                                                   muscle_charuco_ids == i
                                               ].tolist())
                kinematic_camera_imgpts.append(kinematic_charuco_corners[
                                                   kinematic_charuco_ids == i
                                               ].tolist())
                print(f"Found suitable frames ({m_path.name}, {k_path.name})")

            if config["save_images"]:
                save_img_number = str(selected_calib_img_id).zfill(6)
                cv2.imwrite(str(img_out_path / f"muscle_{save_img_number}.jpg"),
                            m_frame)
                cv2.imwrite(str(img_out_path / f"kinematic_{save_img_number}.jpg"),
                            k_frame)
            selected_calib_img_id += 1

assert len(muscle_camera_imgpts) == len(kinematic_camera_imgpts), \
    "Different number of kinematic and muscle points"

# save the points to pkl
calib_pts_out_dir = abs_path / "data/calib_points"
calib_pts_out_dir.mkdir(exist_ok=True)
with open(calib_pts_out_dir/"muscle_pts.pkl", "wb") as f:
    pickle.dump(muscle_camera_imgpts, f)
with open(calib_pts_out_dir/"kinematics_pts.pkl", "wb") as f:
    pickle.dump(kinematic_camera_imgpts, f)









