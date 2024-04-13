from _root_path import add_root 
add_root()

import numpy as np
import open3d as o3d

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.fibers_object as fo

muscles = xu.get_femur_muscles(remove=True)
muscles = fo.Muscles.muscles_from_df(muscles)
muscle = muscles.muscles[0]
type(muscle)

point_cloud = muscle.generate_segment_points(distance=.1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)


vis = o3d.visualization.Visualizer()

vis.create_window()

vis.add_geometry(pcd)

vis.run(); vis.destroy_window()

# voxelization

voxel_size = 1
voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size)

o3d.visualization.draw_geometries([voxel_grid])

# with all linked fibers

point_cloud = muscle.generate_all_linked_segment_points(distance=1.)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

voxel_size = 1
voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size)

o3d.visualization.draw_geometries([voxel_grid])

