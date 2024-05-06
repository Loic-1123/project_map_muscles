from _root_path import add_root
add_root()

import numpy as np
from scipy.spatial.transform import Rotation as R



import map_muscles.muscle_template.map_euler as mp

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

def test_init_Muscle():
    points = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    muscle = mp.Muscle(points)

    def assert_none(*args):
        for arg in args:
            assert arg is None, f"Expected None, got {arg}"

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

# small delta vector

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





if __name__ == '__main__':
    test_compute_alpha()
    test_compute_beta()
    test_compute_gamma()

    test_init_Muscle()
    test_set_axis_points_and_dependant_attributes()
    
    print("test_map_euler.py: All tests passed!")

    