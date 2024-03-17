import cv2 as cv2
from cv2 import aruco
import numpy as np
from tqdm import tqdm


def yield_images_paths(muscle_image_folders, kinematic_image_folders,
                       kin_to_muscle_div_factor, take_n_biggest=1000):
    """
    Every muscle frame/image path has a corresponding kinematic frame path.
    The framerate of the kinematic frames is kin_to_muscle_div_factor times
    the framerate of the muscle images.
    Kinematic frame are saved 3 by 3 in every channel of the an image.

    This function yields the paths of the muscle images and their corresponding
    kinematic images as well as the channel.
    """

    for muscle_folder, kinematic_folder in zip(muscle_image_folders,
                                               kinematic_image_folders):

        #assert kinematic_folder.name in muscle_folder.name, \
        #"The two proposed files are not corresponding"
        kinematic_images_numbers = list(map(lambda x: int(x.stem),
                                            kinematic_folder.glob("*.jpg")))

        min_kinematic_image_index = np.min(kinematic_images_numbers)
        max_kinematic_image_index = np.max(kinematic_images_numbers)

        muscle_images_paths = list(muscle_folder.glob("*.tif"))

        min_muscle_image_index = int(min(muscle_images_paths, key=
                                         lambda x: int(x.stem)).stem)
        muscle_images_paths = sorted(muscle_images_paths,
                                     key=lambda x: x.stat().st_size)[
                              -take_n_biggest:
                              ]

        for muscle_image_path in tqdm(muscle_images_paths):
            searched_kinematic_frame_index = int(muscle_image_path.stem)*\
                                             kin_to_muscle_div_factor + \
                                             min_kinematic_image_index
            search_kinematic_image_channel = searched_kinematic_frame_index % 3 \
                                             - min_kinematic_image_index%3
            search_kinematic_image_index = searched_kinematic_frame_index -\
                                           search_kinematic_image_channel
            kinematic_image_path = kinematic_folder / f"{search_kinematic_image_index}.jpg"

            if kinematic_image_path.exists():
                # Get the index of the muscle frame starting from zero
                muscle_standard_index = (int(muscle_image_path.stem)-min_muscle_image_index)
                # Get the index of the kinematic frame after division
                kinematic_equivalent_index = (searched_kinematic_frame_index-
                         min_kinematic_image_index)/kin_to_muscle_div_factor

                #Check that they match
                assert muscle_standard_index == kinematic_equivalent_index, \
                    "The proposed kinematic and muscle frames are not matching"

                yield (muscle_image_path, kinematic_image_path,
                       search_kinematic_image_channel)

            else:
                print(f"Missing kinematic image {kinematic_image_path} stopping there")
                break


def compare_board_image(image, board, detected_corners_margin=100):
    """
    Display the image focused on the detected board and the board side by side
    so that the user can compare them.
    """
    board_image = board.generateImage((1000, 1000))
    # Lets hope the dictionary is the same at least

    aruco_dict = board.dictionary
    aruco_detector = aruco.ArucoDetector(aruco_dict)
    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(image)
    all_corners = np.hstack(corners)
    # get the most spaced corners
    if len(corners) > 0:
        x_axis = all_corners[:, 0, 0]
        max_x = np.max(x_axis)
        min_x = np.min(x_axis)
        y_axis = all_corners[:, 0, 1]
        max_y = np.max(y_axis)
        min_y = np.min(y_axis)

        max_x = np.max(max_x+detected_corners_margin, image.shape[1])
        min_x = np.min(min_x-detected_corners_margin, 0)
        max_y = np.max(max_y+detected_corners_margin, image.shape[0])
        min_y = np.min(min_y-detected_corners_margin, 0)
        cropped_image = image[min_y:max_y, min_x:max_x]

        cv2.imshow('detected board', cropped_image)
        cv2.imshow('aruco board', board_image)
        cv2.waitKey(0)

    else:
        print("No markers found")
        return

    cv2.imshow('detected board', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

import json
from pathlib import Path
abs_path = Path(__file__).parent
calibration_file = abs_path / 'calibration_parameters.json'
with open(calibration_file, 'r') as f:
    calib = json.load(f)

F = np.array(calib['F'])
e_muscle = np.array((calib['e_muscle']))
e_kin = np.array(calib['e_kin'])
muscle_kin_ratio = np.array(calib['muscle_kin_ratio'])
ref_kin_pts = np.array(calib['ref_kin_pts'])
ref_muscle_pts = np.array(calib['ref_muscle_pts'])


def draw_corresp_points(kin_pts, kin_img, muscle_img, draw_lines=True):
    
    muscle_pts, muscle_line_pts = map_points(kin_pts, muscle_img.shape[:2],
                                             ret_line_pts=True)
    
    kin_img = cv2.cvtColor(kin_img, cv2.COLOR_GRAY2BGR).copy()
    muscle_img = cv2.cvtColor(muscle_img, cv2.COLOR_GRAY2BGR).copy()

    for kin_pt, muscle_pt, line_pt in zip(kin_pts, muscle_pts, muscle_line_pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        kin_img = cv2.circle(kin_img, tuple(kin_pt), 10, color, -1)
        muscle_img = cv2.circle(muscle_img, tuple(muscle_pt), 10, color, 2)
        
        if draw_lines:
            muscle_img = cv2.line(muscle_img,
                                  tuple(line_pt[0]),
                                  tuple(line_pt[1]),
                                  color, 2)
    return kin_img, muscle_img


def map_points(kin_pts, muscle_img_shape, ret_line_pts=False):
    ''' img_line - image on which we draw the epilines for the points in img_pts
        lines - corresponding epilines '''
    
    muscle_lines = cv2.computeCorrespondEpilines(kin_pts.reshape(-1,1,2), 2, F)
    muscle_lines = muscle_lines.reshape(-1,3)

    r, c = muscle_img_shape

    pts_dist = np.linalg.norm(kin_pts-e_kin[:2], axis=-1)
    line_pts_dist = pts_dist * muscle_kin_ratio
    epitope = e_muscle[:2]
    ref_pts = ref_kin_pts
    ref_line_pts = ref_muscle_pts
    
    ref_pts_dist = []
    for pt in kin_pts:
        ref_pts_dist.append(np.linalg.norm(ref_pts-pt, axis=-1))
    ref_pts_dist = np.array(ref_pts_dist)
    assert len(muscle_lines) == len(kin_pts) == len(line_pts_dist)

    muscle_pts = []
    muscle_line_pts = []

    # now find the point on the line that is line_pts_dist away from the epitope
    for r, pt, line_pt_dist, ref_pt_dist_ratio in zip(muscle_lines, kin_pts, line_pts_dist, ref_pts_dist):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        line_vect = np.array([x1-x0, y1-y0])
        line_vect = line_vect/np.linalg.norm(line_vect)
        final_pt_pos = np.round(epitope + line_vect*line_pt_dist).astype(int)
        final_pt_neg = np.round(epitope - line_vect*line_pt_dist).astype(int)

        # Compute the distance to two reference points (if it is the right point, ration should be conserved)
        ref_line_pts_dist_pos = np.linalg.norm(ref_line_pts-final_pt_pos, axis=-1)
        ref_line_pts_dist_neg = np.linalg.norm(ref_line_pts-final_pt_neg, axis=-1)
    
        dist_ratio_pos_std = np.std(ref_line_pts_dist_pos/ref_pt_dist_ratio)
        dist_ratio_neg_std = np.std(ref_line_pts_dist_neg/ref_pt_dist_ratio)
        final_pt = final_pt_pos if dist_ratio_pos_std < dist_ratio_neg_std else final_pt_neg

        muscle_pts.append(final_pt)

        if ret_line_pts:
            muscle_line_pts.append(([x0, y0], [x1, y1]))
    if ret_line_pts:
        muscle_line_pts = np.array(muscle_line_pts)
        return muscle_pts, muscle_line_pts
    else:
        return muscle_pts
