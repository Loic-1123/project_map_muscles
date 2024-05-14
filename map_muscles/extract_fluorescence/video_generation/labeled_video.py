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

import map_muscles.video_converter as vc
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.path_utils as pu

def get_sleap_location_start_id(locations, node_id=0):
    """
    Get the id in the locations array where the labelling starts.
    Assuming that the labelling is continuous and starts after the last Nan value.
    
    Parameters:
        locations (ndarray): The array of locations.
        node_id (int): The ID of the node.
    
    Returns:
        int: The ID where the labelling starts.
    
    """

    node_locations = locations[:, node_id, :, :]
    last_nan_id = np.where(np.isnan(node_locations[:,0]))[0][-1]
    label_start_id = last_nan_id + 1

    return label_start_id

def write_cropped_muscle_frame(
        figsize,
        i, label_start_id,
        trochanter_locations, femur_tibia_locations,
        half_width, margin,
        kin_min_id, kin_to_muscle_div_factor,
        muscle_frames, muscle_min_id, discrepancy,
        video_writer,
        show_labels=False, size_cropped=100,
    ):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    label_id = i + label_start_id

    trochanter_pts = trochanter_locations[label_id, :, 0]
    femur_tibia_pts = femur_tibia_locations[label_id, :, 0]
    pts = np.array([trochanter_pts, femur_tibia_pts])

    # cropped muscle image 
    rectangle = imu.get_rectangle(pts, half_width, margin)

    muscle_id = imu.get_matching_muscle_id(
        kin_frame_id=label_id,
        min_id_kin=kin_min_id,
        kin_to_muscle_div_factor=kin_to_muscle_div_factor,
        min_id_muscle=muscle_min_id
        )
    
    muscle_frame = muscle_frames[muscle_id-(discrepancy//3)]
    muscle_frame = cv2.flip(muscle_frame, 1)
    muscle_pts = imu.map_points(pts, muscle_frame.shape)

    rectangle = imu.get_rectangle(muscle_pts, half_width, margin)
    cropped_muscle_img = imu.get_cropped_rectangle(muscle_frame, rectangle)
    ax.imshow(cropped_muscle_img, cmap='gray')
    ax.set_title(f'Muscle cropped frame {muscle_id}')

    if show_labels:
        pts = imu.get_points_coor_for_cropped_img(muscle_pts, half_width, margin)
        ax.scatter(pts[0][0], pts[0][1], c='g', s=size_cropped, label='Trochanter')
        ax.scatter(pts[1][0], pts[1][1], c='r', s=size_cropped, label='Femur-Tibia')

    ax.axis('off')

    vc.save_frame_plt_film(video_writer, fig)
    plt.close(fig)

def write_cropped_muscle_video(
    label_start_id,
    trochanter_locations, femur_tibia_locations,
    half_width, margin,
    kin_frames, kin_min_id, kin_to_muscle_div_factor,
    muscle_frames, muscle_min_id,
    count_factor,
    video_name, fps, output_dir,
    figsize, discrepancy, n =1,
    show_labels=False,
    ):

    length = min(len(kin_frames), len(trochanter_locations))
    r = range(0, length, n)

    print("===START: Creating video===")
    video_writer = vc.get_video_writer(video_name, figsize, fps, video_dir=output_dir)

    count = 0
    tqdm_r = tqdm.tqdm(r)
    for i in tqdm_r:
        
        if count % count_factor ==0:
            write_cropped_muscle_frame(
                figsize,
                i, label_start_id,
                trochanter_locations, femur_tibia_locations,
                half_width, margin,
                kin_min_id, kin_to_muscle_div_factor,
                muscle_frames, muscle_min_id, discrepancy,
                video_writer=video_writer,
                show_labels=show_labels,
            )
 
        count += 1

    vc.end_cv2_writing(video_writer)
    print("===END: Creating video===")

def write_kin_and_cropped_kin_video(
        kin_frames,
        trochanter_locations, femur_tibia_locations,
        figsize, fps, 
        half_width, margin,
        video_name, output_dir,
        n=1, s=5, s_cropped=100,
        ):
    
    video_writer= vc.get_video_writer(video_name, figsize, fps, video_dir=output_dir)

    length = min(len(kin_frames), len(trochanter_locations))

    r = range(0, length, n)
    print("===START: Creating video===")
    tqdm_r = tqdm.tqdm(r)
    for i in tqdm_r:

        fig, axs = plt.subplots(1, 2, figsize=figsize)
    
        label_id = i + label_start_id

        trochanter_pts = trochanter_locations[label_id, :, 0]
        femur_tibia_pts = femur_tibia_locations[label_id, :, 0]
        pts = np.array([trochanter_pts, femur_tibia_pts])

        img_rect = imu.draw_rectangle(pts, kin_frames[i], half_width, margin)
        axs[0].imshow(img_rect, cmap='gray')

        axs[0].scatter(trochanter_pts[0], trochanter_pts[1], s=s, c='r', label='trochanter')
        axs[0].scatter(femur_tibia_pts[0], femur_tibia_pts[1], s=s, c='g', label='femur-tibia')    
        
        axs[0].set_title(f'Frame {i+kin_label_start_id}')

        # cropped image 
        rectangle = imu.get_rectangle(pts, half_width, margin)
        cropped_img = imu.get_cropped_rectangle(kin_frames[i],rectangle)
        axs[1].imshow(cropped_img, cmap='gray')
        axs[1].set_title(f'Cropped frame {i+kin_label_start_id}')

        # plot points on the cropped image
        cropped_tro, cropped_ft  = imu.get_points_coor_for_cropped_img(pts, half_width, margin)
        axs[1].scatter(cropped_tro[0], cropped_tro[1], s=s_cropped, c='r', label='trochanter')
        axs[1].scatter(cropped_ft[0], cropped_ft[1], s=s_cropped, c='g', label='femur-tibia')

        for ax in axs:
            ax.legend()
            ax.axis('off')

        vc.save_frame_plt_film(video_writer, fig)

        plt.close(fig)

    vc.end_cv2_writing(video_writer)
    print("===END: Creating video===")

def write_cropped_femur_calibration_check_frame(
        figsize,
        i, label_start_id,
        trochanter_locations, femur_tibia_locations,
        half_width, margin,
        kin_min_id, muscle_min_id,
        muscle_frames, discrepancy,
        video_writer,
        s=4, s_cropped=100,
    ):

    fig, axs = plt.subplots(1, 2, figsize=figsize)  
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
    axs[0].scatter(cropped_tro[0], cropped_tro[1], s=s, c='r', label='trochanter')
    axs[0].scatter(cropped_ft[0], cropped_ft[1], s=s, c='g', label='femur-tibia')


    muscle_id = imu.get_matching_muscle_id(
        kin_frame_id=label_id,
        min_id_kin=kin_min_id,
        kin_to_muscle_div_factor=4,
        min_id_muscle=muscle_min_id)
    
    muscle_frame = muscle_frames[muscle_id-(discrepancy//3)]
    muscle_frame = cv2.flip(muscle_frame, 1)
    muscle_pts = imu.map_points(pts, muscle_frame.shape)

    rectangle = imu.get_rectangle(muscle_pts, half_width, margin)
    cropped_muscle_img = imu.get_cropped_rectangle(muscle_frame, rectangle)
    axs[1].imshow(cropped_muscle_img, cmap='gray')
    axs[1].set_title(f'Muscle cropped frame {muscle_id}')

    # plot points on the muscle cropped image
    cropped_tro, cropped_ft  = imu.get_points_coor_for_cropped_img(muscle_pts, half_width, margin)
    axs[1].scatter(cropped_tro[0], cropped_tro[1], s=s_cropped, c='r', label='trochanter')
    axs[1].scatter(cropped_ft[0], cropped_ft[1], s=s_cropped, c='g', label='femur-tibia')


    for ax in axs:
        ax.legend()
        ax.axis('off')

    vc.save_frame_plt_film(video_writer, fig)

    plt.close(fig)




def write_cropped_femur_calibration_check_video(
        video_name,
        trochanter_locations, femur_tibia_locations,
        kin_frames, kin_min_id, label_start_id,
        muscle_frames, muscle_min_id,
        half_width=30, margin=30, 
        fps=3, n=1, count_factor=4, discrepancy=4,
        figsize=(20,10),
        s=4, s_cropped=100,
        out_dir= pu.get_video_dir(),
):
    video_writer = vc.get_video_writer(video_name, figsize, fps, out_dir)

    length = min(len(kin_frames), len(trochanter_locations))
    r = range(0, length, n)

    print("===START: Creating video===")

    count = 0
    tqdm_r = tqdm.tqdm(r)
    for i in tqdm_r:
        if count % count_factor:
            write_cropped_femur_calibration_check_frame(
                figsize,
                i, label_start_id,
                trochanter_locations, femur_tibia_locations,
                half_width, margin,
                kin_min_id, muscle_min_id,
                muscle_frames, discrepancy=discrepancy,
                video_writer=video_writer,
                s=s, s_cropped=s_cropped,
            )

        count += 1

    vc.end_cv2_writing(video_writer)
    print("===END: Creating video===")


if __name__ == "__main__":

    filename = 'label_femur_V1.000_900_1440_kin.analysis.h5'
    sleap_folder = pu.get_sleap_dir()
    file_path = sleap_folder / filename

    kin_dir_path = pu.get_kin_dir(number='900_1440')
    muscle_dir_path = pu.get_muscle_dir(number='900_1440')

    with h5py.File(file_path, 'r') as f:
        locations = f["tracks"][:].T

    TROCHANTER_ID = 0
    FEMUR_TIBIA_ID = 1

    trochanter_locations = locations[:, TROCHANTER_ID, :, :]
    femur_tibia_locations = locations[:, FEMUR_TIBIA_ID, :, :]

    # get id in trochanter_locations where Nan stops (when the labelling starts)
    label_start_id = get_sleap_location_start_id(locations, node_id=TROCHANTER_ID)
    kin_min_id = imu.get_min_id(kin_dir_path, 'jpg')

    # discrepancy between kin and muscle frames
    # it's hard to determine if there is a discrepancy, 
    # but it seems that a discrepancy of 4 frame is fine 
    # (which correspond to a 1 frame discrepancy in the muscle frames)
    discrepancy = 4

    kin_label_start_id = kin_min_id + label_start_id - discrepancy
    kin_label_end_id = len(trochanter_locations) + kin_min_id - discrepancy

    kin_frames = vc.extract_kin_frames(
        kin_dir_path, 'jpg', 
        start_index = kin_label_start_id, 
        end_index = kin_label_end_id
        )

    # plot cropped femur for calibration check
    half_width = 30
    margin = 30

    figsize = (10,15)

    muscle_min_id = imu.get_min_id(muscle_dir_path, 'tif', id_format='{:06d}')
    muscle_frames = vc.extract_muscle_frames(muscle_dir_path, 'tif', start_index = muscle_min_id, gain=1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to use for the video
    output_dir = pu.get_video_dir()
    video_name = 'cropped_femur_muscle_fps2.mp4'

    """
    write_cropped_muscle_video(
        label_start_id,
        trochanter_locations,
        femur_tibia_locations,
        half_width,
        margin,
        kin_frames, kin_min_id, 4,
        muscle_frames, muscle_min_id,
        count_factor=4,
        video_name=video_name, fps=2, output_dir=output_dir,
        figsize=figsize, discrepancy=discrepancy, n =1,
        show_labels=False,
        )
    print("Video created to ", output_dir / video_name)

    video_name = 'cropped_femur_muscle_fps2_with_labels.mp4'
    write_cropped_muscle_video(
        label_start_id,
        trochanter_locations,
        femur_tibia_locations,
        half_width,
        margin,
        kin_frames, kin_min_id, 4,
        muscle_frames, muscle_min_id,
        count_factor=4,
        video_name=video_name, fps=2, output_dir=output_dir,
        figsize=figsize, discrepancy=discrepancy, n=1,
        show_labels=True,
        )
    print("Video created to ", output_dir / video_name)

    video_name = 'labelled_kin_and_cropped_kin_fps6.mp4'
    write_kin_and_cropped_kin_video(
        kin_frames,
        trochanter_locations, femur_tibia_locations,
        figsize=(15,7), fps=6,
        half_width=30, margin=20,
        video_name=video_name, output_dir=output_dir,
    )
    print("Video created to ", output_dir / video_name)

    """
    video_name = 'calibration_check_on_cropped_femur_fps_3.mp4'

    write_cropped_femur_calibration_check_video(
        video_name,
        trochanter_locations, femur_tibia_locations,
        kin_frames, kin_min_id, label_start_id,
        muscle_frames, muscle_min_id,
        half_width=30, margin=30, 
        fps=3, n=1, count_factor=8, discrepancy=discrepancy,
        figsize=(20,10),
        s=100, s_cropped=100,
        out_dir= pu.get_video_dir(),
    )

