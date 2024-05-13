from _root_path import add_root
add_root()

import map_muscles.path_utils as pu
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.video_converter as vc

import cv2

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import tqdm

def write_kin_muscle_matching_clip(
        muscle_ids,
        muscle_dir_path,
        muscle_min_id,
        ratio,
        kin_min_id,
        kin_dir_path,
        video_writer,
        video_name,
        figsize=(10, 5),
        ):
    """
    Writes a video clip showing the matching between muscle images and kin images.

    Args:
        muscle_ids (list): List of muscle IDs.
        muscle_dir (str): Directory path where muscle images are stored.
        muscle_min_id (int): Minimum muscle ID.
        ratio (float): Ratio between images: ratio between the number of images
         taken by the kinetc camera, while the muscle camera took 1 image.
        kin_min_id (int): Minimum kin ID.
        kin_path (str): Path to the kin images.
        video_writer: Video writer object for writing the video clip.
        video_name (str): Name of the output video file.
        figsize (tuple, optional): Figure size for displaying muscle and kin images. Defaults to (10, 5).
    """

    print("===Creating video===")
    tqdm_ids = tqdm.tqdm(muscle_ids)

    for muscle_id in tqdm_ids:

        muscle_img = cv2.imread(str(muscle_dir_path / f'{muscle_id:06d}.tif'), -1)
        kin_img = imu.get_matching_kin_img(
            muscle_id, 
            muscle_min_id,
            ratio, 
            min_id_kin_file=kin_min_id, 
            kin_path=kin_dir_path)
        
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

        vc.save_frame_plt_film(video_writer, fig)

        plt.close(fig)

    print("Video writing complete to ", output_folder / video_name)
    # close video writer
    vc.end_cv2_writing(video_writer)

    print("===END: Creating video===")
    
if __name__ == '__main__':
    kin_path = pu.get_kin_dir()
    muscle_path = pu.get_muscle_dir()

    fps = 30

    start_sec = 15
    end_sec = 40

    kin_min_id = imu.get_min_id(kin_path, 'jpg')
    muscle_min_id = imu.get_min_id(muscle_path, 'tif', id_format='{:06d}')

    kin_start_id = kin_min_id + start_sec * fps
    kin_end_id = kin_min_id + end_sec * fps

    ratio = 4

    muscle_start_id = imu.get_matching_muscle_id(kin_start_id, kin_min_id, ratio, muscle_min_id)
    muscle_end_id = imu.get_matching_muscle_id(kin_end_id, kin_min_id, ratio, muscle_min_id)

    muscle_ids = np.arange(muscle_start_id, muscle_end_id)

    output_folder = pu.get_video_dir()
    video_name = 'testing_kin_muscle_camera_matching.mp4'

    figsize = (10, 5)
    out_fps = 8
    out = vc.get_video_writer(video_name, figsize, out_fps)

    write_kin_muscle_matching_clip(
        muscle_ids,
        muscle_path,
        muscle_min_id,
        ratio,
        kin_min_id,
        kin_path,
        out,
        video_name,
        figsize=figsize,
    )

