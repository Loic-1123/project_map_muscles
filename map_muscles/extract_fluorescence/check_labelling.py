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
import map_muscles.video_converter as vc

filename = 'label_femur_V1.000_900_1440_kin.analysis.h5'

root_path = Path(get_root_path())
sleap_folder = root_path / 'map_muscles' / 'sleap'

file_path = sleap_folder / filename

assert file_path.exists(), f"File not found: {file_path}"

with h5py.File(file_path, 'r') as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

"""
print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()
"""

TROCHANTER_ID = 0
FEMUR_TIBIA_ID = 1
TIBIA_TARSUS_ID = 2

trochanter_locations = locations[:, TROCHANTER_ID, :, :]
femur_tibia_locations = locations[:, FEMUR_TIBIA_ID, :, :]
tibia_tarsus_locations = locations[:, TIBIA_TARSUS_ID, :, :]

mpl.rcParams['figure.figsize'] = [15,6]

kin_path = imu.get_kin_folder()

kin_frames = vc.extract_kin_frames(kin_path, 'jpg')

kin_min_id = imu.get_min_id(kin_path, 'jpg')

len_max = min(len(kin_frames), len(trochanter_locations))

out_path = vc.get_video_dir()
assert out_path.exists(), f"Following out_path does not exist: {out_path}"

figsize = (10, 5)

#video_name = 'check_labelling_on_kinematic_fps6.mp4'
video_name = 'test2.mp4'
fps = 12
out = vc.get_video_writer(video_name, figsize, fps=fps)

frame_start = 710
frame_end = len_max
n = 1

r = range(frame_start, frame_end, n)
print("===Creating video===")

size = 5

for i in tqdm.tqdm(r):
    frame = kin_frames[i]

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.imshow(frame, cmap='gray')
    ax.scatter(trochanter_locations[i, 0, 0], trochanter_locations[i, 1, 0], c='r', s=size, label='trochanter')
    ax.scatter(femur_tibia_locations[i, 0, 0], femur_tibia_locations[i, 1, 0], c='g', s=size, label='femur-tibia')
    ax.scatter(tibia_tarsus_locations[i, 0, 0], tibia_tarsus_locations[i, 1, 0], c='b', s=size, label='tibia-tarsus')
    ax.set_title(f'frame {i}')
    ax.legend()

    # remove axis
    ax.axis('off')

    vc.save_frame_plt_film(out, fig)

    # close figure
    plt.close(fig)

vc.end_cv2_writing(out)

print('Video created to ', out_path)