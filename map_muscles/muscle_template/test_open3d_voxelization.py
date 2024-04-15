from _root_path import add_root
add_root()

import numpy as np
import open3d as o3d

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.fibers_object as fo
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.open3d_voxelization as vx

np.random.seed(0)

muscles = xu.get_femur_muscles(remove=True)
muscles = fo.Muscles.muscles_from_df(muscles).muscles
muscle = muscles[0]
colors = vf.get_random_color_map(muscles)
colors = colors[:, :-1]

def test_muscle_fibers_voxelization(distance=1.0):
    point_cloud = muscle.generate_segment_points(distance=distance)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    voxel_size = 1
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size)

    o3d.visualization.draw_geometries([voxel_grid])

def test_muscles_voxelization(distance=1.0, voxel_size=1.0, show_pre_voxelization=True):
    """
    Test the voxelization of muscles.
    Might have to wait a bit for the second visualization to be processed.

    Parameters:
    - distance (float): The distance between the generated segment points.

    Returns:
    None
    """

    vis = o3d.visualization.Visualizer()

    colors = vf.get_random_color_map(muscles)
    # remove last coordinate of colors
    colors = colors[:, :-1]


    if show_pre_voxelization:

        vis.create_window()

        for m, c in zip(muscles, colors):
            point_cloud = m.generate_segment_points(distance=distance)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            pcd.paint_uniform_color(c)

            vis.add_geometry(pcd)

        vis.run(); vis.destroy_window()

    # voxelization

    point_cloud = muscle.generate_all_linked_segment_points(distance=distance)

    pcd = o3d.geometry.PointCloud()
    
    vis.create_window()

    for m, c in zip(muscles,colors):
        point_cloud = m.generate_all_linked_segment_points(distance=distance)
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.paint_uniform_color(c)
        voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size)

        vis.add_geometry(voxel_grid)

    vis.run(); vis.destroy_window()

def test_basic_muscles_fibers_convex_hull(distance=1.0, voxel_size=1.0, voxelization=False):
    """
    Test the voxelization of muscles.
    Might have to wait a bit for the second visualization to be processed.

    Parameters:
    - distance (float): The distance between the generated segment points.

    Returns:
    None
    """

    vis = o3d.visualization.Visualizer()

    vis.create_window()

    # voxelization
    pcd = o3d.geometry.PointCloud()
    
    for m, c in zip(muscles,colors):
        points = m.generate_segment_points(distance=distance)
        pcd.points = o3d.utility.Vector3dVector(points)

        hull, _ = pcd.compute_convex_hull()
        hull.paint_uniform_color(c)

        if voxelization:
            voxel_grid=o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull,voxel_size)
            vis.add_geometry(voxel_grid)
        else:
            vis.add_geometry(hull)
        

    vis.run(); vis.destroy_window()    

def test_basic_hull_voxel_grid(distance=1.0, voxel_size=1.0):
    grid = vx.basic_hull_voxel_grid(muscle, distance=distance, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([grid])

def test_muscles_basic_hull_voxel_grid(distance=1.0, voxel_size=1.0):

    muscles_grids = []

    for m in muscles:
        grid = vx.basic_hull_voxel_grid(m, distance=distance, voxel_size=voxel_size)
        muscles_grids.append(grid)
    
    vis = o3d.visualization.Visualizer()

    vis.create_window()

    for grid in muscles_grids:
        vis.add_geometry(grid)

    vis.run(); vis.destroy_window()

def test_voxel_grid_to_pcd():
    grid = vx.basic_hull_voxel_grid(muscle, distance=1.0, voxel_size=1.0)
    pcd = vx.voxel_grid_to_pcd(grid)
    o3d.visualization.draw_geometries([pcd])

def test_muscles_hull_pcds(distance=1.0, voxel_size=1.0):
    pcds = vx.muscles_hulls_pcds(muscles, distance=distance, voxel_size=voxel_size)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcd  in pcds:
        vis.add_geometry(pcd)

    vis.run(); vis.destroy_window()
        


if __name__ == "__main__":
    #test_muscle_fibers_voxelizatieon()
    
    #test_muscles_voxelization(voxel_size=10., show_pre_voxelization=False)
    #test_muscles_voxelization(voxel_size=5., show_pre_voxelization=False)
    #test_muscles_voxelization(voxel_size=2., show_pre_voxelization=False)
    #test_muscles_voxelization(voxel_size=1., show_pre_voxelization=False)

    #test_basic_muscles_fibers_convex_hull(voxel_size=1.0, voxelization=True)

    #test_basic_hull_voxel_grid()

    #test_muscles_basic_hull_voxel_grid()

    #test_voxel_grid_to_pcd()

    test_muscles_hull_pcds()











