from _root_path import add_root, get_root_path
add_root()

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import map_muscles.muscle_template.xray_utils as xu


import tqdm 

def get_equally_spaced_colors(n:int, cmap='hsv', rm_alpha_channel=True):
    """
    Generate n equally spaced colors.

    Parameters:
    - n (int): The number of colors to generate.

    Returns:
    - colors (np.ndarray): The array of colors.
    """

    cmap = plt.get_cmap(cmap)

    colors = np.array([cmap(i/n) for i in range(n)])

    # remove alpha channel
    if rm_alpha_channel:
        colors = colors[:, :-1]

    return colors

def basic_hull_voxel_grid(points, voxel_size=1.0):
    """
    Create a voxel grid from the convex hull of the muscle.

    Parameters:
    - muscle (Muscle): The muscle object.
    - voxel_size (float): The size of the voxel grid.

    Returns:
    None
    """
    
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)

    hull, indices = pcl.compute_convex_hull()

    # voxelization of hull
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull, voxel_size)

    return voxel_grid

def voxel_grid_to_pcd(voxelgrid:o3d.geometry.VoxelGrid):
    array_pcd = np.asarray([voxelgrid.origin + pt.grid_index*voxelgrid.voxel_size for pt in voxelgrid.get_voxels()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array_pcd)

    return pcd

def muscles_hulls_pcds(muscles:list, color=True, voxel_size=1.0):
    """
    Create a list of point clouds from the convex hulls of the muscles.

    Parameters:
    - muscles (list): The list of muscle points.
    - voxel_size (float): The size of the voxel grid.

    Returns:
    - pcds (list): The list of point cloud objects.
    """

    pcds = []
    tqdm_muscles = tqdm.tqdm(muscles, desc='Creating point clouds from muscle hulls')
    for muscle in tqdm_muscles:
        muscle_points = points_from_muscle_df(muscle)
        voxel_grid = basic_hull_voxel_grid(muscle_points, voxel_size=voxel_size)
        pcd = voxel_grid_to_pcd(voxel_grid)
        pcds.append(pcd)

    if color:
        colors = get_equally_spaced_colors(len(pcds))
        
        for i, pcd in enumerate(pcds):
            pcd.paint_uniform_color(colors[i])

    return pcds


MUSCLE_NAMES = [
    'LH_FeTi_flexor', 
    'LH_FeTi_anterior_acc_flexor', 
    'LH_FeTi_posterior_acc_flexor', 
    'LH_FeTi_extensor', 
    'LH_FeCl_ltm2',
]

ORDER_IDX = [f'id_{i}' for i in range(1, 6)]

def points_from_muscle_df(muscle:pd.DataFrame):

    pointsA = muscle['pointA'].to_numpy()
    pointsB = muscle['pointB'].to_numpy()
    points = np.concatenate((np.array(pointsA), np.array(pointsB)), axis=0)
    points = np.array([eval(point) for point in points])
    return points

def generate_and_save_3d_muscles_map(
    dir_path: Path, root_name:str, muscles:list, voxel_size=1.0, 
    muscle_names:list=MUSCLE_NAMES, idx=ORDER_IDX
    ):
    
    dir_path.mkdir(exist_ok=True, parents=True)

    muscles_points = []

    for muscle in muscles:
        points = points_from_muscle_df(muscle)
        muscles_points.append(points)
    
    pcds = muscles_hulls_pcds(muscles_points, voxel_size=voxel_size)

    for pcds, muscle_name, indice in zip(pcds, muscle_names, idx):
        file_name = f'{indice}_{root_name}_{muscle_name}'
        file_path = dir_path / file_name

        points = np.asarray(pcds.points)

        np.save(file_path, points)

def load_muscles_map_pcds(dir_path: Path,check_correct_nb:int=5):
    
    files = list(dir_path.glob('*.npy'))
    n = len(files)

    if check_correct_nb:
        assert n==check_correct_nb, f'Expected {check_correct_nb} files, got {n}'
    
    pcds = [o3d.geometry.PointCloud() for _ in range(len(files))]

    for file_path, pcd in zip(files, pcds):
        points = np.load(file_path)
        pcd.points = o3d.utility.Vector3dVector(points)

    print(f'Loaded files: from {dir_path}', files)
        
    return pcds

    
def color_pcds(pcds:list):
    colors = get_equally_spaced_colors(len(pcds))
        
    for i, pcd in enumerate(pcds):
        pcd.paint_uniform_color(colors[i])

    return pcds

def vs_pcds(pcds:list):
    vs = o3d.visualization.Visualizer()
    vs.create_window()

    for pcd in pcds:
        vs.add_geometry(pcd)

    vs.run(); vs.destroy_window()


if __name__ == "__main__":

    df = xu.get_femur_muscles(remove=True)
    #muscles = fo.Muscles.muscles_from_df(df).muscles


    

    data_dir_path = Path(get_root_path()) / 'map_muscles' / 'data' / 'muscles_maps'
    
    dir_name='basic_map_without_fiber_segmentation'

    save_dir_path = data_dir_path / dir_name
    save_dir_path.mkdir(exist_ok=True, parents=True)

    root_name = 'basic_vsize_1.0'

    generate_and_save_3d_muscles_map(save_dir_path, root_name,muscles=df, voxel_size=1.0)

    pcds = load_muscles_map_pcds(save_dir_path)

    pcds = color_pcds(pcds)

    vs_pcds(pcds)