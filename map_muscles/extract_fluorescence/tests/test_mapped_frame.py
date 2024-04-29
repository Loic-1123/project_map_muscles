from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt

import map_muscles.path_utils as pu
import map_muscles.extract_fluorescence.mapped_frame as mf
import map_muscles.muscle_template.map as mp
import map_muscles.extract_fluorescence.plot_map_on_pixel_img as pmp
import map_muscles.extract_fluorescence.imaging_utils as imu

sleap_dir = pu.get_sleap_dir()
sleap_file = 'label_femur_V1.000_900_1440_kin.analysis.h5'
sleap_file_path = sleap_dir / sleap_file

tro, ft, tt = pmp.get_locations_from_sleap_file(sleap_file_path)

idx = len(tro) - 100

tro_point = tro[idx]
fem_tib_point =ft[idx]
IMGAXIS = np.array([tro_point, fem_tib_point])

kin_dir = pu.get_kin_dir()
kin_min = imu.get_min_id(kin_dir, 'jpg')

kin_frames_dir = pu.get_kin_frames_dir()
kin_file = 'kin_frames_3601_5800.npy'
kin_file_path = kin_frames_dir/ kin_file

KINFRAME = pmp.get_labeled_kin_frame(kin_file_path, idx)

map_dir = pu.get_basic_map_dir()
MMAP = mp.MuscleMap.from_directory(map_dir)
MMAP.set_default_axis_points()

FIGSIZE = (10, 10)

def test_plot_coordinate_frame():
    mframe = mf.MappedFrame(KINFRAME, IMGAXIS, MMAP)

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_img(ax, cmap='gray')
    mframe.plot_coordinate_frame(ax)

    ax.axis('off')
    ax.legend()

    plt.show()


def get_scaled_translated_map():
    ratio = pmp.compute_ratio_between_map_and_kin_frame(MMAP, tro_point, fem_tib_point)
    mmap = pmp.scaled_translate_map(MMAP, IMGAXIS, ratio)
    return mmap

def test_zero_yaw_pitch_mmap_axis():

    mmap = get_scaled_translated_map()
    mmap = mmap.to_yaw(0).to_pitch(0)
    
    mframe = mf.MappedFrame(KINFRAME, IMGAXIS, mmap)

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_img(ax, cmap='gray')
    mframe.plot_map_axis(ax)
    origin = mframe.get_mmap().get_map2d().get_axis_points()[0]+(0,-10)
    mframe.plot_coordinate_frame(ax, origin=origin)

    ax.axis('off')
    ax.legend()

    plt.show()

def test_zero_yaw_pitch_mmap():
    mmap = get_scaled_translated_map()
    mmap = mmap.to_yaw(0).to_pitch(0)

    mframe = mf.MappedFrame(KINFRAME, IMGAXIS, mmap)

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_img(ax, cmap='gray')
    mframe.plot_map_on_frame(ax, s=1)
    origin = mframe.get_mmap().get_map2d().get_axis_points()[0]+(-10,-20)
    mframe.plot_coordinate_frame(ax, origin=origin)

    ax.axis('off')
    ax.legend()

    plt.show()

DELTA_ORIGIN = (-10,-10)

def test_yaw_pitch_mmap_axis(yaw=np.pi/2, pitch=0, delta_origin=DELTA_ORIGIN):
    mmap = get_scaled_translated_map()
    mmap = mmap.to_pitch(pitch).to_yaw(yaw)
    
    mframe = mf.MappedFrame(KINFRAME, IMGAXIS, mmap)

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_img(ax, cmap='gray')
    mframe.plot_map_axis(ax)
    origin = mframe.get_mmap().get_map2d().get_axis_points()[0]+delta_origin
    mframe.plot_coordinate_frame(ax, origin=origin)

    ax.axis('off')
    ax.legend()

    plt.show()

def test_yaw_angles(yaws, delta_origin=DELTA_ORIGIN):
    mmap = get_scaled_translated_map().to_pitch(0)
    mframe= mf.MappedFrame(KINFRAME, IMGAXIS, mmap)

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)
    mframe.plot_img(ax, cmap='gray')
    origin = mframe.get_mmap().get_map2d().get_axis_points()[0]+delta_origin
    mframe.plot_coordinate_frame(ax, origin=origin)

    colors = mp.get_equally_spaced_colors(len(yaws))

    for yaw, color in zip(yaws, colors):
        mframe = mf.MappedFrame(KINFRAME, IMGAXIS, mmap.to_yaw(yaw))
        mframe.plot_map_axis(ax, label=f'yaw: {yaw:.2f}', color=color)

    ax.axis('off')
    ax.legend()

    plt.show()
        


if __name__ == '__main__':
    #test_plot_coordinate_frame()
    #test_zero_yaw_pitch_mmap_axis()
    #test_zero_yaw_pitch_mmap()
    
    yaw = np.pi/2
    pitch = 0
    test_yaw_pitch_mmap_axis(yaw, pitch)

    yaws = [
        0,
        np.pi/4,
        np.pi/2,
        3*np.pi/4,
        np.pi,
        3*np.pi/2,             
    ]

    #test_yaw_angles(yaws)
    






