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

images_path = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording' / '900_1440'
kin_path = images_path / 'kin'

kin_frames = vc.extract_kin_frames(kin_path, 'jpg')

kin_min_id = imu.get_min_id(kin_path, 'jpg')

len_max = min(len(kin_frames), len(trochanter_locations))

out_path = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording' / 'videos'
assert out_path.exists(), f"Following out_path does not exist: {out_path}"

width = 1000
height = 500
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to use for the video

video_name = 'kin_labelled.mp4'
output_file = out_path / video_name
out = cv2.VideoWriter(str(output_file), fourcc, 2, (width, height))  # Adjust width and height accordingly

frame_start = 710
frame_end = len_max
n = 1

r = range(frame_start, frame_end, n)

tqdm_range = tqdm.tqdm(r)

print("===Creating video===")

size = 2

for i in tqdm_range:
    frame = kin_frames[i]

    fig, ax = plt.subplots(1,1, figsize=(10,5))
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

cv2.destroyAllWindows()
out.release()


print('Video created to ', out_path)