from _root_path import add_root, get_root_path
add_root()

from pathlib import Path

import h5py
import numpy as np
import cv2
import tqdm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import map_muscles.video_converter as vc
import map_muscles.extract_fluorescence.imaging_utils as imu


### PATHS ###
root_path = Path(get_root_path())

filename = 'label_femur_V1.000_900_1440_kin.analysis.h5'
sleap_folder = root_path / 'map_muscles' / 'sleap'
file_path = sleap_folder / filename
assert file_path.exists(), f"File not found: {file_path}"

data_path = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording' / '900_1440'

kin_path = data_path / 'kin'
muscle_path = data_path / 'muscle'

### ###

### LOAD DATA ###

with h5py.File(file_path, 'r') as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

TROCHANTER_ID = 0
FEMUR_TIBIA_ID = 1
TIBIA_TARSUS_ID = 2

trochanter_locations = locations[:, TROCHANTER_ID, :, :]
femur_tibia_locations = locations[:, FEMUR_TIBIA_ID, :, :]

### ###

# get id in trochanter_locations where Nan stops (when the labelling starts)
last_nan_id = np.where(np.isnan(trochanter_locations[:, 0]))[0][-1]
label_start_id = last_nan_id + 1
kin_min_id = imu.get_min_id(kin_path, 'jpg')

discrepancy = 4

kin_label_start_id = kin_min_id + label_start_id - discrepancy
kin_label_end_id = len(trochanter_locations) + kin_min_id - discrepancy

kin_frames = vc.extract_kin_frames(kin_path, 'jpg', start_index = kin_label_start_id, end_index = kin_label_end_id)


# plot some frames with labels to check correspondence
size = 4
size_cropped = 100
half_width = 30
margin = 20

figzise = (25,10)
factor = 100
video_dimensions = (int(figzise[0]*factor), int(figzise[1]*factor))
"""
fourcc = vc.get_fourcc()
video_name = 'labelled_kinematic_and_cropped_femur.mp4'
output_file = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording' / 'videos' / video_name
fps = 10
out = cv2.VideoWriter(str(output_file), fourcc, fps, video_dimensions)  # Adjust width and height accordingly

if not out.isOpened():
    print("Error: Failed to open video writer.")
    exit()

n = 1
length = min(len(kin_frames), len(trochanter_locations))
r = range(0, length, n)
print(f"Creating {video_name} with {length} frames...")
for i in tqdm.tqdm(r):
    fig, axs = plt.subplots(1, 2, figsize=figzise)
    
    # frame 4356 see to adjust: label_id = ... + 1 or + 2? let's take + 1
    label_id = i + label_start_id

    trochanter_pts = trochanter_locations[label_id, :, 0]
    femur_tibia_pts = femur_tibia_locations[label_id, :, 0]
    pts = np.array([trochanter_pts, femur_tibia_pts])

    img_rect = imu.draw_rectangle(pts, kin_frames[i], half_width, margin)
    axs[0].imshow(img_rect, cmap='gray')

    axs[0].scatter(trochanter_pts[0], trochanter_pts[1], s=size, c='r', label='trochanter')
    axs[0].scatter(femur_tibia_pts[0], femur_tibia_pts[1], s=size, c='g', label='femur-tibia')    
    
    axs[0].set_title(f'Frame {i+kin_label_start_id}')

    # cropped image 
    rectangle = imu.get_rectangle(pts, half_width, margin)
    cropped_img = imu.get_cropped_rectangle(kin_frames[i],rectangle)
    axs[1].imshow(cropped_img, cmap='gray')
    axs[1].set_title(f'Cropped frame {i+kin_label_start_id}')

    # plot points on the cropped image
    cropped_tro, cropped_ft  = imu.get_points_coor_for_cropped_img(pts, half_width, margin)
    axs[1].scatter(cropped_tro[0], cropped_tro[1], s=size_cropped, c='r', label='trochanter')
    axs[1].scatter(cropped_ft[0], cropped_ft[1], s=size_cropped, c='g', label='femur-tibia')

    for ax in axs:
        ax.legend()
        ax.axis('off')

    vc.save_frame_plt_film(out, fig)

    plt.close(fig)


cv2.destroyAllWindows()
out.release()

print("Video created to ", output_file)
"""
# plot cropped femur for calibration check
size = 4
size_cropped = 100
half_width = 30
margin = 30

figzise = (25,10)
factor = 100

# video_dimensions = 10 times the size of the figure
video_dimensions = (int(figzise[0]*factor), int(figzise[1]*factor))

muscle_min_id = imu.get_min_id(muscle_path, 'tif', id_format='{:06d}')
muscle_frames = vc.extract_muscle_frames(muscle_path, 'tif', start_index = muscle_min_id, gain=1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to use for the video
video_name = 'calibration_check_on_cropped_femur.mp4'
output_file = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording' / 'videos' / video_name
fps =6
out = cv2.VideoWriter(str(output_file), fourcc, fps, video_dimensions)  

if not out.isOpened():
    print("Error: Failed to open video writer.")
    exit()

n = 1
length = min(len(kin_frames), len(trochanter_locations))
r = range(0, length, n)
print(f"Creating {video_name} with {length} frames...")
for i in tqdm.tqdm(r):
    fig, axs = plt.subplots(1, 2, figsize=figzise)
    
    # frame 4356 see to adjust: label_id = ... + 1 or + 2? let's take + 1
    label_id = i + label_start_id

    trochanter_pts = trochanter_locations[label_id, :, 0]
    femur_tibia_pts = femur_tibia_locations[label_id, :, 0]
    pts = np.array([trochanter_pts, femur_tibia_pts])

    # cropped image 
    rectangle = imu.get_rectangle(pts, half_width, margin)
    cropped_img = imu.get_cropped_rectangle(kin_frames[i],rectangle)
    axs[0].imshow(cropped_img, cmap='gray')
    axs[0].set_title(f'Kinetic cropped frame {i+kin_label_start_id}')
    # plot points on the kin cropped image
    cropped_tro, cropped_ft  = imu.get_points_coor_for_cropped_img(pts, half_width, margin)
    axs[0].scatter(cropped_tro[0], cropped_tro[1], s=size_cropped, c='r', label='trochanter')
    axs[0].scatter(cropped_ft[0], cropped_ft[1], s=size_cropped, c='g', label='femur-tibia')


    muscle_id = imu.get_matching_muscle_id(
        kin_frame_id=label_id,
        min_id_kin=kin_min_id,
        kin_to_muscle_div_factor=4,
        min_id_muscle=muscle_min_id)
    
    muscle_frame = muscle_frames[muscle_id-(discrepancy//2)]
    muscle_frame = cv2.flip(muscle_frame, 1)
    muscle_pts = imu.map_points(pts, muscle_frame.shape)

    rectangle = imu.get_rectangle(muscle_pts, half_width, margin)
    cropped_muscle_img = imu.get_cropped_rectangle(muscle_frame, rectangle)
    axs[1].imshow(cropped_muscle_img, cmap='gray')
    axs[1].set_title(f'Muscle cropped frame {muscle_id}')

    # plot points on the muscle cropped image
    cropped_tro, cropped_ft  = imu.get_points_coor_for_cropped_img(muscle_pts, half_width, margin)
    axs[1].scatter(cropped_tro[0], cropped_tro[1], s=size_cropped, c='r', label='trochanter')
    axs[1].scatter(cropped_ft[0], cropped_ft[1], s=size_cropped, c='g', label='femur-tibia')


    for ax in axs:
        ax.legend()
        ax.axis('off')

    vc.save_frame_plt_film(out, fig)

    plt.close(fig)


cv2.destroyAllWindows()
out.release()

print("Video created to ", output_file)











