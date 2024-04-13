from _root_path import add_root 
add_root()

import numpy as np
import open3d as o3d

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.fibers_object as fo



def basic_hull_voxel_grid(muscle: fo.Fibers, distance=1.0, voxel_size=1.0):
    """
    Create a voxel grid from the convex hull of the muscle.

    Parameters:
    - muscle (Muscle): The muscle object.
    - distance (float): The distance between the generated segment points.
    - voxel_size (float): The size of the voxel grid.

    Returns:
    None
    """
    segments_points = muscle.generate_segment_points(distance=distance)
    

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(segments_points)

    hull, indices = pcl.compute_convex_hull()

    # voxelization of hull
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull, voxel_size)

    return voxel_grid

def voxel_grid_to_pcd(voxelgrid:o3d.geometry.VoxelGrid):
    """
    Convert a voxel grid to a point cloud.

    Parameters:
    - voxelgrid (VoxelGrid): The voxel grid object.

    Returns:
    - pcd (PointCloud): The point cloud object.
    """

    voxels = voxelgrid.get_voxels()
    points = []
    for voxel in voxels:
        points.append(voxel.grid_index)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def muscles_hulls_pcds(muscles:list, random_color=True, distance=1.0, voxel_size=1.0):
    """
    Create a list of point clouds from the convex hulls of the muscles.

    Parameters:
    - muscles (list): The list of muscle objects.
    - distance (float): The distance between the generated segment points.
    - voxel_size (float): The size of the voxel grid.

    Returns:
    - pcds (list): The list of point cloud objects.
    """

    pcds = []
    for muscle in muscles:
        voxel_grid = basic_hull_voxel_grid(muscle, distance=distance, voxel_size=voxel_size)
        pcd = voxel_grid_to_pcd(voxel_grid)
        pcds.append(pcd)

    if random_color:
        colors = vf.get_random_color_map(pcds, rgb_only=True)
        
        for i, pcd in enumerate(pcds):
            pcd.paint_uniform_color(colors[i])

    return pcds


if __name__ == "__main__":

    df = xu.get_femur_muscles(remove=True)
    muscles = fo.Muscles.muscles_from_df(df).muscles
    muscle = muscles[0]

    voxel_grid = basic_hull_voxel_grid(muscle)
    voxels = voxel_grid.get_voxels()
    points = []
    for voxel in voxels:
        points.append(voxel.grid_index)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd])
