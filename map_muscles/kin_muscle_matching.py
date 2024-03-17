from _root_path import add_root, get_root_path
add_root()
import map_muscles.extract_fluorescence.imaging_utils as imu


from pathlib import Path
import cv2

import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

###

root_path = Path(get_root_path())
map_muscles_dir_path = root_path / 'map_muscles'    


data_dir = 'data'
recording_dir = '20240213_muscle_recording'
image_dir = '900_1440'

image_dir_path = map_muscles_dir_path / data_dir / recording_dir / image_dir

kin_path = image_dir_path / 'kin'
muscle_path = image_dir_path / 'muscle'

assert kin_path.exists(), f'Following kin_path does not exist: {kin_path}'
assert muscle_path.exists(), f'Following muscle_path does not exist: {muscle_path}'

###

muscle_files = imu.get_biggest_files(muscle_path, n_files=100)


# for file in muscle_files: print(file); print(file.stat().st_size)
selected_muscle_file = muscle_files[-1] #biggest file
muscle_img = cv2.imread(str(selected_muscle_file), cv2.IMREAD_GRAYSCALE)

###
plt.figure()
plt.imshow(muscle_img, cmap='gray')
plt.title('Muscle Image')
plt.show()

'''
###
import yaml
calibration_config_path = map_muscles_dir_path / 'calibration' / 'config' /'calib_config.yaml'
with open(calibration_config_path, 'r') as file:
    config = yaml.safe_load(file)
kin_to_muscle_div_factor = config['kin_to_muscle_div_factor']

# find min_id_muscle_file
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
    



min_id_muscle_file = get_min_id(muscle_path, 'tif', id_format='{:06d}')
min_id_kin_file = get_min_id(kin_path, 'jpg', id_format=None)

###

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

def get_matching_muscle_id(kin_frame_id:int, min_id_kin:int, kin_to_muscle_div_factor:float, min_id_muscle:int):
    
    muscle_frame_id = ((kin_frame_id - min_id_kin)/ kin_to_muscle_div_factor) + min_id_muscle
    return muscle_frame_id

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

###

# plot muscle and kin image
muscle_image = cv2.imread(str(selected_muscle_file), -1) # -1 important
muscle_frame_id = int(selected_muscle_file.stem)
kin_img = get_matching_kin_img(
    muscle_frame_id, 
    min_id_muscle_file,
    kin_to_muscle_div_factor, 
    min_id_kin_file, kin_path)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(muscle_image, cmap='gray')
ax[0].set_title('Muscle Image')
ax[1].imshow(kin_img, cmap='gray')
ax[1].set_title('Kin Image')

###

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

#cropped_muscle_image = cropped_image(muscle_image, plot=True)
#plt.figure()
#plt.imshow(cropped_muscle_image, cmap='gray')

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

###

cropped_imgs = cropped_images(muscle_image, margin=70, plot=True)

fig, ax = plt.subplots(1, len(cropped_imgs), figsize=(10, 5))
for i, cropped_img in enumerate(cropped_imgs):
    ax[i].imshow(cropped_img, cmap='gray')
    ax[i].set_title(f'Cropped Muscle Image {i+1}')

plt.show()

###

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


selected_muscle_file = muscle_files[-1]
muscle_image = cv2.imread(str(selected_muscle_file), -1) # -1 important
muscle_frame_id = int(selected_muscle_file.stem)
kin_img = get_matching_kin_img(
    muscle_frame_id, 
    min_id_muscle_file,
    kin_to_muscle_div_factor, 
    min_id_kin_file, kin_path)

cropped_imgs = cropped_images(muscle_image, margin=70, plot=True)

kin_img_shape = kin_img.shape
kin_crop_margin = 300
kin_image_centers = (kin_img_shape[1]//2, kin_img_shape[0]//2)
cropped_kin_image = kin_img[kin_image_centers[1] - kin_crop_margin:kin_image_centers[1] + kin_crop_margin,
                             kin_image_centers[0] - kin_crop_margin:kin_image_centers[0] + kin_crop_margin]

cropped_muscle_image = ...

#TODO from muscle cropped images, get corresponding cropped kin images

#TODO plot cropped muscle images and their associated kin images

#TODO function for comparing multiple images


# good indexes to show correspondance of muscle and kin images
# -1, 12, 50

'''