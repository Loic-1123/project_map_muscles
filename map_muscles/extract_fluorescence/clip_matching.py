from _root_path import add_root, get_root_path
add_root()

import map_muscles.path_utils as pu
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.video_converter as vc


from pathlib import Path
import cv2

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import tqdm

###

image_dir_path = pu.get_img_dir()
output_folder = pu.get_video_dir()
#video_name = 'kin_muscle_match3.mp4'
video_name = 'test.mp4'

kin_path = pu.get_kin_dir()
muscle_path = pu.get_muscle_dir()

# 24 - 29 sec on 30 fps video

fps = 30

start_sec = 15
end_sec = 40

kin_min_index = imu.get_min_id(kin_path, 'jpg')

kin_start_id = kin_min_index + start_sec * fps

kin_end_id = kin_min_index + end_sec * fps


muscle_min_id = imu.get_min_id(muscle_path, 'tif', id_format='{:06d}')

ratio = 4

start_muscle_index = imu.get_matching_muscle_id(kin_start_id, kin_min_index, ratio, muscle_min_id)
end_muscle_index = imu.get_matching_muscle_id(kin_end_id, kin_min_index, ratio, muscle_min_id)

muscle_ids = np.arange(start_muscle_index, end_muscle_index)

muscle_img = cv2.imread(str(muscle_path / f'{muscle_ids[0]:06d}.tif'), -1)
kin_img = imu.get_matching_kin_img(muscle_ids[0], muscle_min_id, ratio, min_id_kin_file=kin_min_index, kin_path=kin_path)

figsize = (10, 5)
out_fps = 8
out = vc.get_video_writer(video_name, figsize, out_fps)

if not out.isOpened():
    print("Error: Failed to open video writer.")
    exit()

tqdm_ids = tqdm.tqdm(muscle_ids)
for muscle_id in tqdm_ids:

    muscle_img = cv2.imread(str(muscle_path / f'{muscle_id:06d}.tif'), -1)
    kin_img = imu.get_matching_kin_img(
        muscle_id, 
        muscle_min_id,
        ratio, 
        min_id_kin_file=kin_min_index, 
        kin_path=kin_path)
    
    # flip muscle img around x axis
    muscle_img = cv2.flip(muscle_img, 0)

    # then rotate 90 degrees
    muscle_img = cv2.rotate(muscle_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(muscle_img, cmap='gray')
    ax[0].set_title('Muscle Image')
    ax[1].imshow(kin_img, cmap='gray')
    ax[1].set_title('Kin Image')

    # remove axis
    for a in ax:
        a.axis('off')

    vc.save_frame_plt_film(out, fig)

    plt.close(fig)

print("Video writing complete to ", video_file)
# close video writer
cv2.destroyAllWindows()
out.release()

print('Start frame (= sec in): ', start_sec * fps)

# Release resources
out.release()