from _root_path import add_root
add_root()

import numpy as np
from scipy.spatial.transform import Rotation as R



import map_muscles.muscle_template.map_euler as mp

pi = np.pi

def unit(vec:np.ndarray):
    return vec/np.linalg.norm(vec)


def test_compute_alpha():
    def assert_alpha(u, expected):
        alpha = mp.compute_alpha(u)
        mod_alpha = alpha % (2*pi)
        mod_expected = expected % (2*pi)  

        assert np.isclose(mod_alpha, mod_expected, atol=1e-07), \
            f"Expected alpha = {expected}, got {alpha}. \
            mod_alpha={mod_alpha}, mod_expected={mod_expected}. For u = {u}"
    
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
        gamma = mp.compute_gamma(x,y)

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
                    except ValueError as e: pass
                    else: raise AssertionError(f"Expected a ValueError for xr = {xr}: gamma = {expected}. \
                                               Gamma confounded with alpha.")
                else:
                    assert_gamma(xr, yr, expected)


                    


    
    


    

    

    





    






if __name__ == '__main__':
    test_compute_alpha()
    test_compute_beta()
    test_compute_gamma()
    
    print("All tests passed!")

    