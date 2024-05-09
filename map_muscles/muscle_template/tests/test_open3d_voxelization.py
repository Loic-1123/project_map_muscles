from _root_path import add_root
add_root()

import numpy as np
import open3d as o3d

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.open3d_voxelization as vx

np.random.seed(0)

muscles = xu.get_femur_muscles(remove=True)
muscle = muscles[0]
muscle_points = vx.points_from_muscle_df(muscle)
colors = vf.get_random_color_map(muscles)
colors = colors[:, :-1]

def test_basic_hull_voxel_grid(voxel_size=1.0):
    grid = vx.basic_hull_voxel_grid(muscle_points, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([grid])

def test_muscles_basic_hull_voxel_grid(voxel_size=1.0):

    muscles_grids = []

    for m in muscles:
        points = vx.points_from_muscle_df(m)
        grid = vx.basic_hull_voxel_grid(points, voxel_size=voxel_size)
        muscles_grids.append(grid)
    
    vis = o3d.visualization.Visualizer()

    vis.create_window()

    for grid in muscles_grids:
        vis.add_geometry(grid)

    vis.run(); vis.destroy_window()

def test_voxel_grid_to_pcd():
    points = vx.points_from_muscle_df(muscle)
    grid = vx.basic_hull_voxel_grid(points, voxel_size=1.0)
    pcd = vx.voxel_grid_to_pcd(grid)
    o3d.visualization.draw_geometries([pcd])

def test_muscles_hull_pcds(voxel_size=1.0):
    pcds = vx.muscles_hulls_pcds(muscles, voxel_size=voxel_size)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcd  in pcds:
        vis.add_geometry(pcd)

    vis.run(); vis.destroy_window()


if __name__ == "__main__":

    #test_basic_hull_voxel_grid()

    #test_muscles_basic_hull_voxel_grid()

    #test_voxel_grid_to_pcd()

    test_muscles_hull_pcds()











