from _root_path import add_root
add_root()

import h5py
import numpy as np
import matplotlib.pyplot as plt


import map_muscles.path_utils as pu
import map_muscles.video_converter as vc
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.muscle_template.map as mp

def get_locations_from_sleap_file(
        sleap_file_path,
        trochanter_id=0,
        femur_tibia_id=1,
        tibia_tarsus_id=2
        ):
    with h5py.File(sleap_file_path, 'r') as f:
        locations = f["tracks"][:].T

    trochanter_locations = locations[:, trochanter_id, :, 0]
    femur_tibia_locations = locations[:, femur_tibia_id, :, 0]
    tibia_tarsus_locations = locations[:, tibia_tarsus_id, :, 0]

    return trochanter_locations, femur_tibia_locations, tibia_tarsus_locations

def get_labeled_kin_frame(
        npy_file_path,
        idx):
    kin_frames = np.load(npy_file_path)

    labeled_kin_frame = kin_frames[idx]

    return labeled_kin_frame

def compute_ratio_between_map_and_kin_frame(
        mmap,
        tro_point,
        fem_tib_point,
    ):

    axis_points = mmap.get_axis_points()
    map_axis = axis_points[1] - axis_points[0]
    map_axis_norm = np.linalg.norm(map_axis)

    kin_axis = fem_tib_point - tro_point
    kin_axis_norm = np.linalg.norm(kin_axis)

    ratio = kin_axis_norm / map_axis_norm

    return ratio
    

def save_2d_map_on_kin_frame_by_map2d(
        mmap, 
        img_axis_points_2d,
        labeled_kin_frame, 
        out_path=pu.get_map_matching_dir() / 'test_func_2d_map_on_kin_frame.png',
        figsize=(10,5)):
    centered_mmap = mmap.centered_on_axis_point()
    ratio = compute_ratio_between_map_and_kin_frame(centered_mmap, img_axis_points_2d[0], img_axis_points_2d[1])

    centered_map2d = centered_mmap.get_map2d()
    scaled_map = centered_map2d.scale(ratio)

    translated_map = scaled_map.translate(img_axis_points_2d[1])

    translated_points = translated_map.get_points()
    translated_axis = translated_map.get_axis_points()

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(labeled_kin_frame, cmap='gray')
    ax.scatter(translated_points[:,0], translated_points[:,1], c='m', s=10, label='2d map points')
    ax.scatter(tro_point[0], tro_point[1], c='r', s=10, label='Trochanter')
    ax.scatter(fem_tib_point[0], fem_tib_point[1], c='b', s=10, label='Femur-Tibia')
    ax.scatter(translated_axis[:,0], translated_axis[:,1], c='g', s=10, label='Scaled map axis points')

    ax.legend()
    ax.axis('off')

    fig.savefig(out_path)


    return fig, ax

def save_2d_map_on_kin_frame_by_3dmap(
        mmap, 
        img_axis_points_2d,
        labeled_kin_frame, 
        out_path= pu.get_map_matching_dir() / 'test_oo_map_on_kin_frame.png',
        figsize=(10,5)):
    
    ratio = compute_ratio_between_map_and_kin_frame(mmap, img_axis_points_2d[0], img_axis_points_2d[1])

    centered_mmap = mmap.centered_on_axis_point()

    scaled_map = centered_mmap.scale(ratio)

    # 0 as 3th dimension
    img_axis_3d_point = np.array([img_axis_points_2d[1][0], img_axis_points_2d[1][1], 0])
    translated_map = scaled_map.translate(img_axis_3d_point)

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(labeled_kin_frame, cmap='gray')
    ax.scatter(translated_map.get_points()[:,0], translated_map.get_points()[:,1], c='m', s=10, label='2d map points')

    ax.legend()
    ax.axis('off')

    fig.savefig(out_path)

def save_2d_map_on_kin_frame_muscle_maps(
        mmap, 
        img_axis_points_2d,
        labeled_kin_frame, 
        out_path= pu.get_map_matching_dir() / 'test_map_muscles_on_kin_frame.png',
        figsize=(10,5)):
    
    ratio = compute_ratio_between_map_and_kin_frame(mmap, img_axis_points_2d[0], img_axis_points_2d[1])

    centered_mmap = mmap.centered_on_axis_point()

    scaled_map = centered_mmap.scale(ratio)

    # 0 as 3th dimension
    img_axis_3d_point = np.array([img_axis_points_2d[1][0], img_axis_points_2d[1][1], 0])
    translated_map = scaled_map.translate(img_axis_3d_point)

    map2d = translated_map.get_map2d()
    maps = map2d.get_maps()

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(labeled_kin_frame, cmap='gray')
    ax.scatter(tro_point[0], tro_point[1], c='r', s=10, label='Trochanter')
    ax.scatter(fem_tib_point[0], fem_tib_point[1], c='b', s=10, label='Femur-Tibia')

    for m in maps:
        pts = m.get_points()
        ax.scatter(pts[:,0], pts[:,1], s=1, label=m.get_name())

    ax.legend()
    ax.axis('off')

    fig.savefig(out_path)

def save_individual_maps_on_kin_frames(
        mmap, 
        img_axis_points_2d,
        labeled_kin_frame, 
        out_dir= pu.get_map_matching_dir(),
        out_root='map_on_kin_frame',
        figsize=(10,5)):
    
    ratio = compute_ratio_between_map_and_kin_frame(mmap, img_axis_points_2d[0], img_axis_points_2d[1])

    centered_mmap = mmap.centered_on_axis_point()

    scaled_map = centered_mmap.scale(ratio)

    # 0 as 3th dimension
    img_axis_3d_point = np.array([img_axis_points_2d[1][0], img_axis_points_2d[1][1], 0])
    translated_map = scaled_map.translate(img_axis_3d_point)

    map2d = translated_map.get_map2d()
    maps = map2d.get_maps()

    for m in maps:
        pts = m.get_points()
        fig, ax = plt.subplots(1,1, figsize=figsize)

        ax.imshow(labeled_kin_frame, cmap='gray')
        ax.scatter(tro_point[0], tro_point[1], c='r', s=10, label='Trochanter')
        ax.scatter(fem_tib_point[0], fem_tib_point[1], c='b', s=10, label='Femur-Tibia')

        ax.scatter(pts[:,0], pts[:,1], s=1, label=m.get_name())

        ax.legend()
        ax.axis('off')

        im_file_name = f'{out_root}_{m.get_name()}.png'
        out_path = out_dir / im_file_name

        fig.savefig(out_path)

def scaled_translate_map(mmap, img_axis_points_2d, ratio):
    centered_mmap = mmap.centered_on_axis_point()
    scaled_map = centered_mmap.scale(ratio)

    # 0 as 3th dimension
    img_axis_3d_point = np.array([img_axis_points_2d[1][0], img_axis_points_2d[1][1], 0])
    translated_map = scaled_map.translate(img_axis_3d_point)

    return translated_map

def save_map_with_yaw_pitch(
        mmap,
        yaw,
        pitch,
        img_axis_points_2d,
        labeled_kin_frame,
        out_path= pu.get_map_matching_dir() / 'map_on_kin_frame_yaw_pitch.png',
        figsize=(10,5)
        ):
    
    ratio = compute_ratio_between_map_and_kin_frame(mmap, img_axis_points_2d[0], img_axis_points_2d[1])

    mmap = mmap.to_yaw(yaw).to_pitch(pitch)

    translated_map = scaled_translate_map(mmap, img_axis_points_2d, ratio)

    map2d = translated_map.get_map2d()

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(labeled_kin_frame, cmap='gray')

    pts = map2d.get_points()

    ax.scatter(pts[:,0], pts[:,1], s=1, label='Map points')

    ax.legend()
    ax.axis('off')

    fig.savefig(out_path)

    
def save_map_yaw_pitch_maps(
        mmap,
        yaw,
        pitch,
        img_axis_points_2d,
        labeled_kin_frame,
        out_path= pu.get_map_matching_dir() / 'maps_on_kin_frame_yaw_pitch.png',
        figsize=(10,5)
        ):
    
    ratio = compute_ratio_between_map_and_kin_frame(mmap, img_axis_points_2d[0], img_axis_points_2d[1])

    mmap = mmap.to_yaw(yaw).to_pitch(pitch)

    translated_map = scaled_translate_map(mmap, img_axis_points_2d, ratio)

    map2d = translated_map.get_map2d()

    maps = map2d.get_maps()

    fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.imshow(labeled_kin_frame, cmap='gray')

    for m in maps:
        pts = m.get_points()
        ax.scatter(pts[:,0], pts[:,1], s=1, label=m.get_name())

    axis = map2d.get_axis_points()
    ax.plot(axis[:,0], axis[:,1], c='g', label='Axis')

    ax.legend()
    ax.axis('off')

    fig.savefig(out_path)





    
    

if __name__ == "__main__":


    sleap_dir = pu.get_sleap_dir()
    sleap_file = 'label_femur_V1.000_900_1440_kin.analysis.h5'
    sleap_file_path = sleap_dir / sleap_file

    tro, ft, tt = get_locations_from_sleap_file(sleap_file_path)

    kin_dir = pu.get_kin_dir()
    kin_min = imu.get_min_id(kin_dir, 'jpg')

    kin_frames_dir = pu.get_kin_frames_dir()
    kin_file = 'kin_frames_3601_5800.npy'
    kin_file_path = kin_frames_dir/ kin_file

    idx = len(tro) - 100

    labeled_kin_frame = get_labeled_kin_frame(kin_file_path, idx)


    tro_point = tro[idx]
    fem_tib_point =ft[idx]

    map_dir = pu.get_basic_map_dir()
    mmap = mp.MuscleMap.from_directory(map_dir)
    mmap.set_default_axis_points()

    ratio = compute_ratio_between_map_and_kin_frame(mmap, tro_point, fem_tib_point)

    img_axis = np.array([tro_point, fem_tib_point])

    #save_2d_map_on_kin_frame_by_map2d(mmap=mmap,img_axis_points_2d=img_axis,labeled_kin_frame=labeled_kin_frame)   

    #save_2d_map_on_kin_frame_by_3dmap(mmap=mmap,img_axis_points_2d=img_axis,labeled_kin_frame=labeled_kin_frame)

    #save_2d_map_on_kin_frame_muscle_maps(mmap=mmap,img_axis_points_2d=img_axis,labeled_kin_frame=labeled_kin_frame)

    #save_individual_maps_on_kin_frames(mmap=mmap,img_axis_points_2d=img_axis,labeled_kin_frame=labeled_kin_frame)

    #set pitch and yaw

    

    #save_map_with_yaw_pitch(mmap, yaw, pitch, img_axis, labeled_kin_frame)


    """
    pitch = (np.pi/2)*0.8
    yaw = (np.pi)

    save_map_yaw_pitch_maps(mmap, yaw, pitch, img_axis, labeled_kin_frame)
    # -> pitch is yaw and yaw is pitch: correction needed
    """

    pitch = np.pi * 1
    yaw = np.pi * 0.
    #save_map_yaw_pitch_maps(mmap, yaw, pitch, img_axis, labeled_kin_frame)







    

