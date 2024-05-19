from _root_path import add_root
add_root()

import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


import map_muscles.path_utils as pu
import map_muscles.muscle_template.euler_map as mp
from map_muscles.muscle_template.tests.test_euler_map_Muscle import add_coor_frame, add_pcd_xy_plane


pi = np.pi

def test_map_constructor():

    dir_path = pu.get_basic_map_dir()

    mmap = mp.MuscleMap(dir_path)

    assert mmap.get_name() == "MuscleMap"
    assert mmap.get_axis_points() == None
    assert mmap.get_angles() == (None, None, None)

def get_map():
    dir_path = pu.get_basic_map_dir()
    mmap = mp.MuscleMap.from_directory(dir_path)
    mmap.init_default_axis_points()
    return mmap

def test_visualize_map():
    mmap = get_map()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_default(vis)

    vis.run(); vis.destroy_window()

def array_colors(color:np.array, n):
    return [color for i in range(n)]


def test_visualize_translation():
    mmap = get_map()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_default(vis)

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    dist = 500

    # translations
    n = len(mmap.get_muscles())
    mmap1 = mmap.translate(x * dist)
    mmap1.draw_points(vis, colors=array_colors([1, 0, 0], n))
    mmap2 = mmap.translate(y * dist)
    mmap2.draw_points(vis, colors = array_colors([0, 1, 0], n))
    mmap3 = mmap.translate(z * dist)
    mmap3.draw_points(vis, colors = array_colors([0, 0, 1], n))
    mmap4 = mmap.translate((x+y+z) * dist)
    mmap4.draw_points(vis, colors = array_colors([0.5, 0.5, 0.5],n))

    center = mmap.get_axis_points()[0]
    add_coor_frame(vis, center, size=300)
    add_pcd_xy_plane(vis, 2000, 100, z=center[2], center=(center[0], center[1]))
    

    vis.run(); vis.destroy_window()

def test_visualize_reset_rotation():
    mmap = get_map()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_default(vis)

    r_mmap = mmap.reset_rotation()
    r_mmap.draw_points(vis, colors = array_colors([0, 0, 0], len(mmap.get_muscles())))
    r_mmap.draw_axis(vis, color = [0, 0, 0])
    r_mmap.draw_xyz_vectors(vis, length=1000)

    add_coor_frame(vis, mmap.get_axis_points()[0], size=300)
    add_pcd_xy_plane(vis, 2000, 100, z=mmap.get_axis_points()[0][2], center=(mmap.get_axis_points()[0][0], mmap.get_axis_points()[0][1]))

    vis.run(); vis.destroy_window()



def test_visualize_rotation():
    mmap = get_map()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_default(vis)

    n = len(mmap.get_muscles())
    # rotations

    angles1 = np.array([3*pi/4, pi/4, 0]) # should be point to the tetrahedron last point with the frame, and be rotated (roll)
    angles2 = np.array([pi/2, 0, pi/2])
    angles3 = np.array([pi, 0, 0])
    # 2 and 3 should be confounded, and lying on the z axis
    angles4 = np.array([0, pi/2, pi/2]) # should be lying on the x-y plane, (z_vec= [0,-1,0])
    # some random angles
    np.random.seed(0)
    angles5 = np.random.rand(3)*2*pi

    mmap1 = mmap.rotate_to_angles(angles1)
    mmap1.draw_points(vis, colors = array_colors([1, 0.5, 0],n))
    mmap.draw_axis(vis, color = [1, 0.5, 0])

    mmap2 = mmap.rotate_to_angles(angles2)
    mmap2.draw_points(vis, colors = array_colors([0, 0.5, 1],n))
    mmap.draw_axis(vis, color = [0, 0.5, 1])
    mmap3 = mmap.rotate_to_angles(angles3)
    mmap3.draw_points(vis, colors = array_colors([0.5, 1, 0],n))
    mmap.draw_axis(vis, color = [0.5, 1, 0])

    mmap4 = mmap.rotate_to_angles(angles4)
    mmap4.draw_points(vis, colors = array_colors([0.5, 0, 1],n))
    mmap.draw_axis(vis, color = [0.5, 0, 1])

    mmap5 = mmap.rotate_to_angles(angles5)
    mmap5.draw_points(vis, colors = array_colors([0.7, 0.7, 0.7],n))
    mmap.draw_axis(vis, color = [0.7, 0.7, 0.7])

    center = mmap.get_axis_points()[0]
    add_coor_frame(vis, center, size=300)
    add_pcd_xy_plane(vis, 2000, 100, z=center[2], center=(center[0], center[1]))
    

    vis.run(); vis.destroy_window()
    
def test_visualize_roll():
    mmap = get_map()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_default(vis)

    n = 9

    dist = 250

    # rolls

    for i in range(n):
        gamma = i*2*pi/n
        d = i * dist
        angles = mmap.get_angles()
        angles = (angles[0], angles[1], gamma)
        mmap1 = mmap.rotate_to_angles(angles).translate(np.array([d, 0, 0]))
        mmap1.draw_default(vis)

    center = mmap.get_axis_points()[0]
    add_coor_frame(vis, center, size=500)
    

    vis.run(); vis.destroy_window()








if __name__ == "__main__":
    test_map_constructor()
    #test_visualize_map()
    #test_visualize_translation()
    #test_visualize_reset_rotation()
    #test_visualize_rotation()
    #test_visualize_roll()
    
    print("All tests passed.")