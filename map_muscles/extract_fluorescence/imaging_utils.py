from _root_path import add_root, get_root_path
add_root()

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import find_peaks

import json


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

def get_biggest_files(files_path, n_files=100, extension='tif'):
    """
    Get the n_files biggest files in a directory
    """

    files = sorted(files_path.glob(f"*.{extension}"), key=lambda x: x.stat().st_size)[-n_files:]
    return files

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
        kin_frame_id, 
        kin_frame_channel):
    
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
    
    kin_frame_channel = get_matching_kin_frame_channel(kin_frame_id, min_id_kin_file)

    kin_image_id = get_matching_kin_image_id(
        min_id_kin_file,
        kin_frame_id, 
        kin_frame_channel
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


        



### ###
def get_index_from_frame_id(frame_id, min_frame_id, min_img_id):
    ...

    #TODO
    


def cropped_image(muscle_image, margin=100, width=10, prominence=0.9, plot=False):
    mean_x = np.mean(muscle_image, axis=0)
    mean_y = np.mean(muscle_image, axis=1)

    x_peaks, x_peaks_descr = find_peaks(mean_x, width=width, prominence=prominence)
    y_peaks, y_peaks_descr = find_peaks(mean_y, width=width, prominence=prominence)

    x_width = x_peaks_descr["widths"]
    y_width = y_peaks_descr["widths"]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(mean_x, label="mean x")
        plt.plot(mean_y, label="mean y")
        plt.scatter(x_peaks, mean_x[x_peaks], c="r", label="x peaks")
        plt.scatter(y_peaks, mean_y[y_peaks], c="g", label="y peaks")
        plt.legend()
    
    max_y_peak = np.argmax(y_peaks_descr["prominences"])
    central_x_peak = np.argmin(np.abs(x_peaks- mean_x.shape[0]/2))

    x_start = int(x_peaks[central_x_peak] - x_width[central_x_peak]) - margin
    x_end = int(x_peaks[central_x_peak] + x_width[central_x_peak]) + margin

    y_start = int(y_peaks[max_y_peak] - y_width[max_y_peak]) - margin
    y_end = int(y_peaks[max_y_peak] + y_width[max_y_peak]) + margin

    cropped_img = muscle_image[y_start:y_end, x_start:x_end]

    return cropped_img


def cropped_images(
        muscle_image, 
        nb_peaks:int=2, 
        margin:int=100, 
        width:int=10, 
        prominence:float=0.9, 
        sort:bool=True,
        plot:bool=False
        ):
    
    """
    Returns the cropped images of the muscle image based on the peaks of the mean x and y
    axis of the muscle image.

    Args:
        muscle_image: The muscle image to be cropped.
        nb_peaks: The number of peaks to consider.
        margin: The margin to be added to the cropped images.
        width: The width of the peaks to be considered.
        prominence: The prominence of the peaks to be considered.
        sort: If True, sort the cropped images by x position.
        plot: If True, plot information about the peaks and the selected areas.

    Returns:
        List of cropped images.
    """

    mean_x = np.mean(muscle_image, axis=0)
    mean_y = np.mean(muscle_image, axis=1)

    x_peaks, x_peaks_descr = find_peaks(mean_x, width=width, prominence=prominence)
    y_peaks, y_peaks_descr = find_peaks(mean_y, width=width, prominence=prominence)

    x_width = x_peaks_descr["widths"]
    y_width = y_peaks_descr["widths"]

    max_y_peak = np.argmax(y_peaks_descr["prominences"])
    sorted_x_peaks = np.argsort(x_peaks_descr["prominences"])
    max_x_peaks = sorted_x_peaks[-nb_peaks:]

    x_starts = []
    x_ends = []
    cropped_imgs = []

    y_start = int(y_peaks[max_y_peak] - y_width[max_y_peak]) - margin
    y_end = int(y_peaks[max_y_peak] + y_width[max_y_peak]) + margin

    for max_x_peak in max_x_peaks:
        x_starts.append(int(x_peaks[max_x_peak] - x_width[max_x_peak]) - margin)
        x_ends.append(int(x_peaks[max_x_peak] + x_width[max_x_peak]) + margin)

    # sort by x position
    if sort: x_starts, x_ends = zip(*sorted(zip(x_starts, x_ends)))

    for x_start, x_end in zip(x_starts, x_ends):
        cropped_imgs.append(muscle_image[y_start:y_end, x_start:x_end])

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(mean_x, label="mean x")
        plt.plot(mean_y, label="mean y")
        plt.scatter(x_peaks, mean_x[x_peaks], c="r", label="x peaks")
        plt.scatter(y_peaks, mean_y[y_peaks], c="g", label="y peaks")
        
        # chosen peaks x and y
        size=100
        plt.scatter(x_peaks[max_x_peaks], mean_x[x_peaks[max_x_peaks]], 
                    c="k", label="chosen peaks", marker="x", s=size)
        plt.scatter(y_peaks[max_y_peak], mean_y[y_peaks[max_y_peak]], 
                    c="k", marker="x", s=size) 

        # plot segments at the bottom of the plot to show selected area
        bottom = min(min(mean_y), min(mean_x))
        plt.plot([x_starts[0], x_ends[0]], [bottom, bottom], c="b", linestyle="-", label="x selected ranges")
        for x_start, x_end in zip(x_starts, x_ends):
            plt.plot([x_start, x_end], [bottom, bottom], c="b", linestyle="-")
            bottom = bottom - 3
                
        plt.plot([y_start, y_end], [bottom, bottom], linestyle="-", label="y selected range")

        plt.legend()
    
    return cropped_imgs


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
