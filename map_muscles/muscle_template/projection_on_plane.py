import numpy as np
import matplotlib.pyplot as plt
from skspatial.objects import Points, Plane

def assert_orthogonalilty(U):
    for u1 in U:
        for u2 in U:
            if u1 is u2:
                continue
            assert np.isclose(np.dot(u1, u2), 0), \
            f"Not orthogonal: u1: {u1}, u2: {u2}, dot: {np.dot(u1, u2)}"

def project_point_on_plane(point, U):    
    projection = 0
    for u in U:
        projection += (np.dot(point, u) * u)/np.dot(u, u)

    return projection

def plane_abcd(v1, v2, plot_function=True):
    cp  = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, v1)

    if plot_function:
        plotting = lambda xx, yy : (-a*xx -b*yy + d)/c
        return a, b, c, d, plotting
    else:
        return a, b, c, d

def project_segment_on_plane(segment, U):
    return np.array([project_point_on_plane(point, U) for point in segment])

def projected_segments(segments, U):
    assert_orthogonalilty(U)
    return [project_segment_on_plane(segment, U) for segment in segments]

def orthonormal_vectors(v1, v2):
    v2_on_v1 = np.dot(v2, v1) / np.dot(v1, v1) * v1
    orthogonal_vector = v2 - v2_on_v1

    # orthogonality
    assert np.isclose(np.dot(orthogonal_vector, v1), 0)

    u1 = v1 / np.linalg.norm(v1)
    u2 = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    # normality
    assert np.isclose(np.dot(u1, u1), 1)
    assert np.isclose(np.dot(u2, u2), 1)

    return u1, u2


if __name__ == "__main__":
    v1 = np.array([1.,1.,1.])
    v2 = np.array([1.,.0,.0])

    fiber_segment = np.array([[3, 5, 1], [7, 4, 3]])
    f1 = fiber_segment[0]
    f2 = fiber_segment[1]

    # project f1 on v1-v2 plane

    ## find orthogonal vector spanning v1-v2 plane
    ### project v2 on v1
    v2_on_v1 = np.dot(v2, v1) / np.dot(v1, v1) * v1

    ### subtract v2_on_v1 from v2 to get orthogonal vector
    orthogonal_vector = v2 - v2_on_v1
    assert np.isclose(np.dot(orthogonal_vector, v1), 0)

    u1 = v1 / np.linalg.norm(v1)
    u2 = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    assert np.isclose(np.dot(u1, u1), 1)
    assert np.isclose(np.dot(u2, u2), 1)

    projected_fiber = project_segment_on_plane(fiber_segment, [u1, u2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # plot v1 and v2
    ax.plot(*np.array([np.zeros(3), v1]).T, color='g')
    ax.plot(*np.array([np.zeros(3), v2]).T, color='r')

    # plot fiber
    ax.plot(*fiber_segment.T, color='b')
    ax.plot(*projected_fiber.T, color='b', linestyle='--')

    # plot plane
    xx, yy = np.meshgrid(range(10), range(10))

    a,b,c,d, plot_plane = plane_abcd(v1, v2)

    ax.plot_surface(xx, yy, plot_plane(xx, yy), alpha=0.5)

    plt.show()
