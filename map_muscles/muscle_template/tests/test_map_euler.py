from _root_path import add_root
add_root()

import numpy as np
import map_muscles.muscle_template.map_euler as mp

pi = np.pi

def unit(vec:np.ndarray):
    return vec/np.linalg.norm(vec)


def test_compute_alpha():
    def assert_alpha(u, expected):
        alpha = mp.compute_alpha(u)
        assert np.isclose(alpha, expected, atol=1e-07), \
            f"Expected alpha = {expected}, got {alpha}. For u = {u}"
    
    # aligned with z-axis -> undifined alpha
    u = np.array([0,0,1])
    try: alpha = mp.compute_alpha(u)
    except AssertionError as e: pass
    else: raise AssertionError("Expected an AssertionError in test_compute_alpha() for u = [0,0,1]")

    u = np.array([0, 0, -1])
    try: alpha = mp.compute_alpha(u)
    except AssertionError as e: pass
    else: raise AssertionError("Expected an AssertionError in test_compute_alpha() for u = [0,0,-1]")

    # not unit vector
    u = np.array([1,1,0])
    try: alpha = mp.compute_alpha(u)
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
        expected = np.pi
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

    
    for z in zs:
        u = np.array([0,0,z])
        expected = 0
        assert_beta(u, expected)

    u = np.array([1,0,0])
    expected = 0
    assert_beta(u, expected)

    





if __name__ == '__main__':
    test_compute_alpha()
    #test_compute_beta()
    
    print("All tests passed!")