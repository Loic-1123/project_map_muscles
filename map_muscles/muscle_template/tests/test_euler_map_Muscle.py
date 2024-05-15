from _root_path import add_root
add_root()

import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from itertools import product


import map_muscles.muscle_template.euler_map as mp
from map_muscles.muscle_template.euler_map import assert_angle
import map_muscles.path_utils as pu

"""
This file contains tests for the class Muscle() in euler_map.py.
"""

pi = np.pi

def unit(vec:np.ndarray):
    return vec/np.linalg.norm(vec)

# z_vector and alpha beta angles ground truth
ZAB = [
    (unit([1,0,0]), pi/2, pi/2),
    (unit([0,1,0]), pi, pi/2),
    # (unit([0,0,1]), None, None), # undefined alpha and gamma, beta = 0
    (unit([1,1,0]), 3*pi/4, pi/2),
    (unit([1,0,1]), pi/2, pi/4),
    (unit([0,1,1]), pi, pi/4),
    
    (unit([1,1,1]), 3*pi/4, np.arccos(1/np.sqrt(3))),
    
    (unit([-1,0,0]), 3*pi/2, pi/2),
    (unit([0,-1,0]), 0, pi/2),
    #(unit([0,0,-1]), 0, 0), # undefined alpha and gamma, beta = pi
    (unit([-1,-1,0]), 2*pi - pi/4, pi/2),
    (unit([-1,0,-1]), 3*pi/2, 3*pi/4),
    (unit([0,-1,-1]), 0, 3*pi/4),
    
    (unit([-1,-1,-1]), 2*pi - pi/4, np.arccos(-1/np.sqrt(3))),
    
    (unit([1,-1,0]), pi/4, pi/2),
    (unit([-1,1,0]), pi + pi/4, pi/2),
    (unit([1,0,-1]), pi/2, 3*pi/4),
    (unit([-1,0,1]), 3*pi/2, pi/4),
    (unit([0,1,-1]), pi, 3*pi/4),
    (unit([0,-1,1]), 0, pi/4),

]

def test_compute_alpha():
    def assert_alpha(u, expected):
        alpha = mp.compute_alpha(u, print_warning=False, raise_warning=True)
        mod_alpha = alpha % (2*pi)
        mod_expected = expected % (2*pi)  

        assert np.isclose(mod_alpha, mod_expected, atol=1e-07), \
            f"Expected alpha = {expected}, got {alpha}. \
            mod_alpha={mod_alpha}, mod_expected={mod_expected}. For u = {u}"
    
    # aligned with z-axis -> undifined alpha
    u = unit([0,0,1])
    try: alpha = mp.compute_alpha(u, print_warning=False, raise_warning=True)
    except Warning as w: pass
    else: raise AssertionError("Expected an Warning in test_compute_alpha() for u = [0,0,1]")

    u = unit([0, 0, -1])
    try: alpha = mp.compute_alpha(u, print_warning=False, raise_warning=True)
    except Warning as w: pass
    else: raise AssertionError("Expected an Warning in test_compute_alpha() for u = [0,0,-1]")

    # not unit vector
    u = np.array([1,1,0])
    try: alpha = mp.compute_alpha(u, print_warning=False, raise_warning=True)
    except AssertionError as e: pass
    else: raise AssertionError("Expected an AssertionError in test_compute_alpha() for u = [1,1,0]")

    # test for remarkable angles

    ## as z should not influence alpha, different values of z are tested
    zs = np.arange(-0.9, 0.9, 0.1)
    
    for z in zs:
        u = unit([1,0,z])
        expected = np.pi/2
        assert_alpha(u, expected)

        u = unit([0,1,z])
        expected = pi
        assert_alpha(u, expected)

        u = unit([1,1,z])
        expected = 3*pi/4
        assert_alpha(u, expected)

        u = unit([-1, 0,z])
        expected = 3*pi/2
        assert_alpha(u, expected)

        u = unit([0, -1, z])
        expected = 0
        assert_alpha(u, expected)

        u = unit([1, -1, z])
        expected = pi/4
        assert_alpha(u, expected)

        u = unit([-1,1,z])
        expected = pi + pi/4
        assert_alpha(u, expected)

        u = unit([-1,-1,z])
        expected = 2*pi - pi/4
        assert_alpha(u, expected)
def test_compute_beta():
    
    def assert_beta(u, expected):
        beta = mp.compute_beta(u)
        assert np.allclose(beta, expected), f"Expected beta = {expected}, got {beta}"

    # not unit vector
    u = np.array([1,1,0])
    try: beta = mp.compute_beta(u)
    except AssertionError as e: pass
    else: raise AssertionError("Expected an AssertionError in test_compute_beta() for u = [1,1,0]")

    # test for remarkable angles
    for i in range(-1,1):
        for j in range(-1,1):
            if i == 0 and j == 0: continue
            u = unit([i,j,0])
            expected = pi/2
            assert_beta(u, expected)
    
    u = unit([0,0,1])
    expected = 0
    assert_beta(u, expected)

    u = unit([1,1,1])
    expected = np.arccos(1/np.sqrt(3))
    assert_beta(u, expected)

    u = unit([1,0,1])
    expected = pi/4
    assert_beta(u, expected)

    u = unit([0,1,1])
    expected = pi/4
    assert_beta(u, expected)

    u = unit([0,1,-1])
    expected = 3*pi/4
    assert_beta(u, expected)

    u = unit([1,0,-1])
    expected = 3*pi/4
    assert_beta(u, expected)
def test_compute_gamma():
    x = np.array([1,0,0])
    y = np.array([0,1,0])

    def assert_gamma(x,y, expected):
        gamma = mp.compute_gamma(x,y,print_warning=False, raise_warning=True)

        mod_gamma = gamma % (2*pi)
        mod_expected = expected % (2*pi)  

        assert np.isclose(mod_gamma, mod_expected) or np.isclose(gamma, expected), f"Expected gamma = {expected}, got {gamma}.\
            mod_gamma={mod_gamma}, mod_expected={mod_expected}. For x = {x}, y = {y}"

    step = 0.3
    step2 = 0.1
    alpha_values = np.arange(0, 2*pi, step)
    beta_values = np.arange(0, pi, step2)
    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in np.arange(0, 2*pi, step):
                rot = R.from_euler('ZXZ', [alpha,beta,gamma], degrees=False)
                expected = gamma
                
                

                xr = rot.apply(x)
                yr = rot.apply(y)
                if np.isclose(yr[2],0):
                    try: assert_gamma(xr, yr, expected)
                    except Warning as w: pass
                    else: raise AssertionError(f"Expected a Warning for xr = {xr}, yr={yr}: gamma = {expected}. \
                                               Gamma confounded with alpha.")
                else:
                    assert_gamma(xr, yr, expected)

def assert_none(*args):
    for arg in args:
        assert arg is None, f"Expected None, got {arg}"

def test_init_Muscle():
    points = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    muscle = mp.Muscle(points)

 
    nones = [
        muscle.get_name(),
        muscle.get_axis_points(),
        muscle.get_alpha(),
        muscle.get_beta(),
        muscle.get_gamma(),
        muscle.get_x_vector(),
        muscle.get_y_vector(),
        muscle.get_z_vector(),
    ]

    assert_none(*nones)

def test_set_axis_points_and_dependant_attributes(delta=1e-6+3.7):
    """
    Test function for setting axis points and dependent attributes of a muscle.
    Dependent attributes: alpha, beta, gamma, x_vector, y_vector, z_vector

    Args:
        delta (float): The delta value used for calculations. Default is 1e-6+3.7.

    Returns:
        None
    """
    points = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    muscle = mp.Muscle(points)

    for truth in ZAB:
        z, alpha, beta = truth

        rot = R.from_euler('ZXZ', [alpha,beta,0], degrees=False)

        axis_points= np.array([
            np.array([0,0,0]), z + z*delta
        ])
    
        muscle.set_axis_points(axis_points)

        xr = muscle.get_x_vector()
        yr = muscle.get_y_vector()
        zr = muscle.get_z_vector()

        assert np.isclose(muscle.get_alpha()%(2*pi), alpha%(2*pi)), f"Expected alpha = {alpha}, got {muscle.get_alpha()}"
        assert np.isclose(muscle.get_beta(), beta), f"Expected beta = {beta}, got {muscle.get_beta()}"
        assert muscle.get_gamma() == 0, f"Expected gamma = 0, got {muscle.get_gamma()}"

        assert np.allclose(zr, z), f"Expected z = {truth[0]}, got {z}"
        assert np.allclose(rot.apply([1,0,0]), xr), f"Expected x = {rot.apply([1,0,0])}, got {xr}"
        assert np.allclose(rot.apply([0,1,0]), yr), f"Expected y = {rot.apply([0,1,0])}, got {yr}"

def test_load_from_array_file():
    dir_path = pu.get_basic_map_dir()
    file_path = list(dir_path.glob("*.npy"))[0]

    muscle = mp.Muscle.from_array_file(file_path)

    nones = [
        muscle.get_axis_points(),
        muscle.get_alpha(),
        muscle.get_beta(),
        muscle.get_gamma(),
        muscle.get_x_vector(),
        muscle.get_y_vector(),
        muscle.get_z_vector(),
    ]

    assert_none(*nones)

def get_muscle(idx=0):
    dir_path = pu.get_basic_map_dir()
    file_path = list(dir_path.glob("*.npy"))[idx]

    return mp.Muscle.from_array_file(file_path)

def add_coor_frame(vis, origin, size=200):

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    
    vis.add_geometry(frame)
def generate_xy_plane_points(dim:int, n:int, z=0, center=(0,0)):
    x = np.linspace(-dim, dim, n) + center[0]
    y = np.linspace(-dim, dim, n) + center[1]
    
    xx, yy = np.meshgrid(x, y)

    zz = np.ones_like(xx)*z

    points = np.stack([xx, yy, zz], axis=-1).reshape(-1,3)

    return points

def add_pcd_xy_plane(vis, dim:int, n:int, z=0, center=(0,0)):
    points = generate_xy_plane_points(dim, n, z, center)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vis.add_geometry(pcd)


def test_muscle_visualization():

    m = get_muscle()
    m.init_default_axis_points()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    m.draw_points(vis)
    m.draw_axis(vis)
    m.draw_xyz_vectors(vis)

    center = m.get_axis_points()[0]

    add_coor_frame(vis, center, size=200)

    add_pcd_xy_plane(vis, 1000, 100, z=center[2], center=(center[0], center[1]))

    vis.run(); vis.destroy_window()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    m.draw_default(vis)

    vis.run(); vis.destroy_window()


def test_visualize_gamma_initialization():
    m = get_muscle()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    m.draw_points(vis)

    colors = np.array([[1,0,0],[0,1,0],[0,0,1]])

    n = 6

    for i, gamma in enumerate(np.linspace(0, 2*pi, n)):

        cs = i*colors/n 

        m.init_default_axis_points(gamma=gamma)
        print(gamma)
        m.draw_xyz_vectors(vis, colors=cs)

    vis.run(); vis.destroy_window()
    

def test_visualize_translation():
    m = get_muscle()
    m.init_default_axis_points()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    m.draw_default(vis)

    center = m.get_axis_points()[0]

    add_coor_frame(vis, center, size=200)

    d = 300

    mx = m.translate(translation=m.get_x_vector()*d)

    mx.draw_points(vis, color=[1,0,0])
    mx.draw_axis(vis)
    mx.draw_xyz_vectors(vis)

    my = m.translate(translation=m.get_y_vector()*d)

    my.draw_points(vis, color=[0,1,0])
    my.draw_axis(vis)
    my.draw_xyz_vectors(vis)


    mz = m.translate(translation=m.get_z_vector()*d)

    mz.draw_points(vis, color=[0,0,1])
    mz.draw_axis(vis)
    mz.draw_xyz_vectors(vis)

    mt = m.translate(translation=[100,100,100])

    mt.draw_points(vis, color=[1,0.5,0])
    mt.draw_axis(vis)
    mt.draw_xyz_vectors(vis)
    
    vis.run(); vis.destroy_window()


def test_reset_rotation_correct_angles_and_vectors():
    
    m = get_muscle()
    m.init_default_axis_points()

    mr = m.reset_rotation(raise_assertions=True)

    assert mr.get_alpha() == 0, f"Expected alpha = 0, got {mr.get_alpha()}"
    assert mr.get_beta() == 0, f"Expected beta = 0, got {mr.get_beta()}"
    assert mr.get_gamma() == 0, f"Expected gamma = 0, got {mr.get_gamma()}"

    assert np.allclose(mr.get_x_vector(), [1,0,0]), f"Expected x = [1,0,0], got {mr.get_x_vector()}"
    assert np.allclose(mr.get_y_vector(), [0,1,0]), f"Expected y = [0,1,0], got {mr.get_y_vector()}"
    assert np.allclose(mr.get_z_vector(), [0,0,1]), f"Expected z = [0,0,1], got {mr.get_z_vector()}"

    m_point = m.get_axis_points()[0]
    r_point = mr.get_axis_points()[0]

    assert np.allclose(m_point,r_point),\
    f"Expected first axis points to coincide but got m axis point: {m_point}, mr axis point {r_point}"

def test_visualize_reset_rotation():

    """ 
    Reset muscle axis should be aligned with z-axis (blue)
    The first axis point of the muscles should coincide.
    """
    m = get_muscle()
    m.init_default_axis_points()

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    m.draw_default(vis)

    center = m.get_axis_points()[0]

    add_coor_frame(vis, center, size=200)

    mr = m.reset_rotation(raise_assertions=True)

    mr.draw_points(vis, color=[1,0.5,0])
    mr.draw_axis(vis)
    mr.draw_xyz_vectors(vis)

    vis.run(); vis.destroy_window()

def test_rotate_to_angles():
    m = get_muscle()
    m.init_default_axis_points()

    delta=0.1

    alphas = np.linspace(delta, 2*pi-delta, 15)
    betas = np.linspace(0, pi-delta, 7)
    gammas = np.linspace(delta, 2*pi-delta, 9)

    ab = list(product(alphas, betas))
    angles_triplets = list(product(ab, gammas))
    angles_triplets = [(*list(ab), g) for ab, g in angles_triplets]

    for angles in angles_triplets:
        mr = m.rotate_to_angles(angles, raise_assertions=True, print_debug_info=False)

        alpha, beta, gamma = angles

        mr_alpha = mr.get_alpha()
        mr_beta = mr.get_beta()
        mr_gamma = mr.get_gamma()

        angles_str = '\n'+f'alpha={alpha}, beta={beta}, gamma={gamma}' + '\n' + f'mr_alpha={mr_alpha}, mr_beta={mr_beta}, mr_gamma={mr_gamma}'

        if np.isclose(beta, 0) or np.isclose(beta, pi):
            # alpha and gamma are confounded
            assert_str = f"Expected alpha = 0, got {mr_alpha}" + angles_str
            assert_angle(mr_alpha, 0, assert_str=assert_str)
            assert_str = f"Expected gamma = {gamma+alpha}, got {mr_gamma}" + angles_str
            assert_angle(mr_gamma,gamma+alpha, assert_str=assert_str)
        else:
                
            assert_str = f"Expected alpha = {alpha}, got {mr_alpha}" + angles_str
            assert_angle(mr_alpha, alpha, assert_str=assert_str)
            assert_str = f"Expected gamma = {gamma}, got {mr_gamma}"+   angles_str
            assert_angle(mr_gamma, gamma, assert_str=assert_str)

        assert_str = f"Expected beta = {beta}, got {mr_beta}" + angles_str
        assert_angle(beta, mr_beta, assert_str=assert_str)


        rot = R.from_euler('ZXZ', [alpha,beta,gamma], degrees=False)

        xr = rot.apply([1,0,0])
        yr = rot.apply([0,1,0])
        zr = rot.apply([0,0,1])

        assert np.allclose(mr.get_x_vector(), xr), f"Expected x = {xr}, got {mr.get_x_vector()}"
        assert np.allclose(mr.get_y_vector(), yr), f"Expected y = {yr}, got {mr.get_y_vector()}"
        assert np.allclose(mr.get_z_vector(), zr), f"Expected z = {zr}, got {mr.get_z_vector()}"

        m_point = m.get_axis_points()[0]
        r_point = mr.get_axis_points()[0]

        assert np.allclose(m_point,r_point),\
        f"Expected first axis points to coincide but got m axis point: {m_point}, mr axis point {r_point}"


        







    



if __name__ == '__main__':
    test_compute_alpha()
    test_compute_beta()
    test_compute_gamma()

    test_init_Muscle()
    test_set_axis_points_and_dependant_attributes()
    test_load_from_array_file()

    #test_muscle_visualization()
    #test_visualize_gamma_initialization()
    #test_visualize_translation()

    test_reset_rotation_correct_angles_and_vectors()
    
    #test_visualize_reset_rotation()

    test_rotate_to_angles()
    


    
    print("All tests passed! (test_map_euler.py)")

    