from _root_path import add_root
add_root()

import h5py
import numpy as np
import matplotlib.pyplot as plt

import map_muscles.path_utils as pu
import map_muscles.video_converter as vc
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.muscle_template.map as mp

import open3d as o3d

out_dir = pu.get_map_matching_dir()

#points locations from sleap file
sleap_file_name = 'label_femur_V1.000_900_1440_kin.analysis.h5'
sleap_folder = pu.get_sleap_dir()
sleap_file_path = sleap_folder / sleap_file_name
assert sleap_file_path.exists(), f"File not found: {sleap_file_path}"

# ---START: to modularize in another file --- #
with h5py.File(sleap_file_path, 'r') as f:
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

TROCHANTER_ID = 0
FEMUR_TIBIA_ID = 1
TIBIA_TARSUS_ID = 2

trochanter_locations = locations[:, TROCHANTER_ID, :, :]
femur_tibia_locations = locations[:, FEMUR_TIBIA_ID, :, :]
tibia_tarsus_locations = locations[:, TIBIA_TARSUS_ID, :, :]
# ---END: to modularize in another file --- #


# get a labeled kin image with
kin_path = pu.get_kin_dir()
kin_min_id = imu.get_min_id(kin_path, 'jpg')

kin_frames_dir = pu.get_kin_frames_dir()
file_name = 'kin_frames_3601_5800.npy'
file_path = kin_frames_dir / file_name
kin_frames = np.load(file_path)
idx = len(trochanter_locations) - 100

labeled_kin_frame = kin_frames[idx]

tro_point = trochanter_locations[idx,:,0]
fem_tib_point = femur_tibia_locations[idx,:,0]


"""
figsize = (10,5)

fig, ax = plt.subplots(1,1, figsize=figsize)

rect_half_width = 100
margin = 100
pts = np.array([tro_point, fem_tib_point])
img_rect = imu.draw_rectangle(pts, labeled_kin_frame, rectangle_half_width=rect_half_width, margin=margin)

ax.imshow(img_rect, cmap='gray', alpha=0.5)
ax.scatter(tro_point[0], tro_point[1], c='r', s=10, label='Trochanter')
ax.scatter(fem_tib_point[0], fem_tib_point[1], c='b', s=10, label='Femur-Tibia')

ax.legend()
ax.axis('off')

im_file_name = 'labeled_kin_frame.png'
out_path = out_dir / im_file_name
fig.savefig(out_path)

# cropped img

rect = imu.get_rectangle(pts, half_width=rect_half_width, margin=margin)
cropped_img = imu.get_cropped_rectangle(labeled_kin_frame, rect)
cropped_tro, cropped_ft = imu.get_points_coor_for_cropped_img(pts, half_width=rect_half_width, margin=margin)

fig, ax = plt.subplots(1,1, figsize=figsize)

ax.imshow(cropped_img, cmap='gray')
ax.scatter(cropped_tro[0], cropped_tro[1], c='r', s=10, label='Trochanter')
ax.scatter(cropped_ft[0], cropped_ft[1], c='b', s=10, label='Femur-Tibia')

ax.axis('off')

im_file_name = 'cropped_labeled_kin_frame.png'
out_path = out_dir / im_file_name
fig.savefig(out_path)
"""
# get a muscle map
map_dir = pu.get_basic_map_dir()
mmap = mp.MuscleMap.from_directory(map_dir)
com = mmap.get_com()
direction = np.array([
    0.4, 0.6, -0.3
])


#bad_axis_points_approximation
dist = 700
axis_points = np.array([
    com-dist*direction,
    com + dist*direction
])
mmap.set_axis_points(axis_points)

"""
frame_center = com - 600*direction - 200*np.array([1,1,1])
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=frame_center)


vis = o3d.visualization.Visualizer()
vis.create_window()
mmap.draw_default(vis)
vis.add_geometry(frame)

vis.run(); vis.destroy_window()
"""

# get 3d map axis length, get 2d map axis length, compute ratio -> transition from 2d to 3d
mmap_axis_points = mmap.get_axis_points()

#init reference points
centered_mmap = mmap.centered_on_axis_point()

centered_mmap_axis_points = centered_mmap.get_axis_points()
ref_mmap_points = centered_mmap_axis_points[:, :2]

img_axis_points = np.array([tro_point, fem_tib_point])
ref_img_axis_points = img_axis_points - img_axis_points[0]

# get the ratio
length_ref_mmap = np.linalg.norm(ref_mmap_points[1] - ref_mmap_points[0])
length_ref_img = np.linalg.norm(ref_img_axis_points[1] - ref_img_axis_points[0])

ratio = length_ref_img/length_ref_mmap

# plot pixel image, with 2d scaled projection of 3d map, with first axis points aligned (not the axis yet)
centered_map2d = centered_mmap.get_map2d()
scaled_map = centered_map2d.scale(ratio)

translated_map = scaled_map.translate(img_axis_points[0])

translated_points = translated_map.get_points()
translated_axis = translated_map.get_axis_points()

figsize = (10,5)
fig, ax = plt.subplots(1,1, figsize=figsize)

ax.imshow(labeled_kin_frame, cmap='gray')
ax.scatter(translated_points[:,0], translated_points[:,1], c='m', s=10, label='2d map points')
ax.scatter(tro_point[0], tro_point[1], c='r', s=10, label='Trochanter')
ax.scatter(fem_tib_point[0], fem_tib_point[1], c='b', s=10, label='Femur-Tibia')
ax.scatter(translated_axis[:,0], translated_axis[:,1], c='g', s=10, label='Scaled map axis points')



ax.legend()
ax.axis('off')

im_file_name = '2d_map_on_kin_frame.png'
out_path = out_dir / im_file_name

fig.savefig(out_path)


