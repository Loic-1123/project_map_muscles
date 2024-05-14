from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import numpy as np
import cv2
import json


"""
This file contains functions to handle and get muscle and kin images, 
such has getting corresponding indices or images for muscle and kin images.
It also contains functions to map points from kin to muscle images and vice versa,
as well as to crop images based on points.
"""

root_path = Path(get_root_path())
calibration_file = root_path / 'map_muscles' / 'calibration' / 'calibration_parameters.json'

with open(calibration_file, 'r') as f:
    calib = json.load(f)

F = np.array(calib['F'])
e_muscle = np.array((calib['e_muscle']))
e_kin = np.array(calib['e_kin'])
muscle_kin_ratio = np.array(calib['muscle_kin_ratio'])
ref_kin_pts = np.array(calib['ref_kin_pts'])
ref_muscle_pts = np.array(calib['ref_muscle_pts'])

img_path = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording'

assert img_path.exists(), f"Following path for recorded images does not exist: {img_path}"
def get_min_id(files_path, extension, id_format=None, return_str=False):
    files = sorted(files_path.glob(f"*.{extension}"))
    ids = list(map(lambda x: int(x.stem), files))

    min_id = np.min(ids)
    if id_format is not None:
        min_id_str = id_format.format(min_id)
    else:
        min_id_str = str(min_id)
    
    assert (files_path / f'{min_id_str}.{extension}').exists(),\
        f"Following file (minimum id) does not exist: {files_path / f'{min_id_str:06d}.{extension}'}."
    
    if return_str:
        return min_id, min_id_str
    else:
        return min_id


### kin to muslce ###

def get_matching_kin_frame_id(muscle_frame_id, min_id_muscle, kin_to_muscle_div_factor, min_id_kin_file,):
    kin_frame_id = (muscle_frame_id-min_id_muscle) * kin_to_muscle_div_factor + min_id_kin_file
    return kin_frame_id

def get_matching_kin_frame_channel(kin_frame_id, min_id_kin_file):
    kin_frame_channel = kin_frame_id % 3 - min_id_kin_file % 3
    return kin_frame_channel

def get_matching_kin_image_id(
        min_id_kin_file,
        kin_frame_id
        ):
    
    kin_frame_channel = get_matching_kin_frame_channel(kin_frame_id, min_id_kin_file)
    kin_image_id = kin_frame_id - kin_frame_channel

    return kin_image_id

def get_matching_kin_img(
        muscle_frame_id, 
        min_id_muscle_file,
        kin_to_muscle_div_factor,
        min_id_kin_file, 
        kin_path,
        format = None,
        extension='jpg'):
    
    """
    Returns the matching kin image for a given muscle frame.

    Args:
        muscle_frame_id (int): The frame ID of the muscle image.
        min_id_muscle_file (int): The minimum ID of the muscle files.
        kin_to_muscle_div_factor (int/float): The division factor between kin frame ID and muscle frame ID.
        min_id_kin_file (int): The minimum ID of the kin files.
        kin_path (str): The path to the kin image directory.
        format (str, optional): The format string for the kin image ID. Defaults to None.
        extension (str, optional): The file extension of the kin images. Defaults to 'jpg'.

    Returns:
        The matching kin image.
    """
    
    kin_frame_id = get_matching_kin_frame_id(
        muscle_frame_id,
        min_id_muscle_file,
        kin_to_muscle_div_factor, 
        min_id_kin_file
        )

    kin_image_id = get_matching_kin_image_id(
        min_id_kin_file,
        kin_frame_id, 
        )
    
    if format is not None:
        kin_image_id_str = format.format(kin_image_id)
    else:
        kin_image_id_str = str(kin_image_id)

    kin_img_path = kin_path / f"{kin_image_id_str}.{extension}"

    assert kin_img_path.exists(), f"Following kin_img_path does not exist: {kin_img_path}"

    kin_img = cv2.imread(str(kin_img_path), cv2.IMREAD_GRAYSCALE)
    

    return kin_img

### muscle to kin ###

def get_matching_muscle_id(
        kin_frame_id:int, 
        min_id_kin:int, 
        kin_to_muscle_div_factor:float, 
        min_id_muscle:int):
    muscle_frame_id = ((kin_frame_id - min_id_kin)/ kin_to_muscle_div_factor) + min_id_muscle
    return int(muscle_frame_id)

def get_matching_muscle_img(
        muscle_path:Path,
        kin_frame_id:int,
        min_id_kin:int,
        kin_to_muscle_div_factor:float,
        min_id_muscle:int,
        id_format:str='{:06d}',
):
    muscle_frame_id = get_matching_muscle_id(kin_frame_id, min_id_kin, kin_to_muscle_div_factor, min_id_muscle)
    
    if id_format is not None:
        muscle_frame_id_str = id_format.format(muscle_frame_id)
    else:
        muscle_frame_id_str = str(muscle_frame_id)

    muscle_img_path = muscle_path / f"{muscle_frame_id_str}.jpg"

    assert muscle_img_path.exists(), f"Following muscle_img_path does not exist: {muscle_img_path}"

    muscle_img = cv2.imread(str(muscle_img_path), -1)

    return muscle_img

def get_rectangle(pts, half_width, margin):
    pts_vector = pts[1] - pts[0]
    pts_vector = pts_vector/np.linalg.norm(pts_vector)
    perp_pts_vector = np.array([-pts_vector[1], pts_vector[0]])
    perp_pts_vector = perp_pts_vector/np.linalg.norm(perp_pts_vector)

    rectangle_pts = np.array([pts[0] - pts_vector*margin  - half_width*perp_pts_vector,
                                pts[0] - pts_vector*margin + half_width*perp_pts_vector,
                                pts[1] + pts_vector*margin + half_width*perp_pts_vector,
                                pts[1] + pts_vector*margin - half_width*perp_pts_vector]).astype(int)
    return rectangle_pts

def draw_rectangle(pts, img, rectangle_half_width=50, margin=0):
    rectangle_pts = get_rectangle(pts, rectangle_half_width, margin)
    img_rect = cv2.polylines(img.copy(), [rectangle_pts], isClosed=True, color=(255, 255, 255), thickness=2)
    return img_rect

def get_cropped_rectangle(img, rect_pts):
    # Set width and height of output image
    W = np.linalg.norm(rect_pts[1] - rect_pts[0]).round().astype(int)
    H = np.linalg.norm(rect_pts[2] - rect_pts[1]).round().astype(int)

    # Define points in input image: top-left, top-right, bottom-right, bottom-left
    pts0 = rect_pts.astype(np.float32)

    # Define corresponding points in output image
    pts1 = np.float32([[0,0],[W,0],[W,H],[0,H]])

    # Get perspective transform and apply it
    M = cv2.getPerspectiveTransform(pts0, pts1)
    result = cv2.warpPerspective(img, M, (W,H))
    return result

def get_points_coor_for_cropped_img(pts, half_width, margin):
    pts_vector = pts[1] - pts[0]
    length = np.linalg.norm(pts_vector)
    point1 = half_width, margin
    point2 = half_width, margin + length

    return point1, point2

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
