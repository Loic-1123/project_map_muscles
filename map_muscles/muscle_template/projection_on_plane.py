from _root_path import add_root, get_root_path
add_root()

import numpy as np
import matplotlib.pyplot as plt
import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
# To use later: from skspatial.objects import Points, Plane

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
    # doesnt work for vertical planes: c=0

    cp  = np.cross(v1, v2)

    a, b, c = cp
    d = np.dot(cp, v1)

    if plot_function:
        if c!=0:
            plotting = lambda xx, yy : (-a*xx -b*yy + d)/c
        else:
            def vertical_plane(xx, yy):
                zz = np.zeros(xx.shape)
                return zz
            plotting = vertical_plane
        
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


#if __name__ == "__main__":

femur_muscles = xu.get_femur_muscles()

v1 = np.array([1.0,.0,.0])
v2 = np.array([.0,1.0,.0])

# project f1 on v1-v2 plane

## find orthogonal vector spanning v1-v2 plane
### project v2 on v1
u1, u2 = orthonormal_vectors(v1, v2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = vf.get_random_color_map(femur_muscles)

for muscle, c in zip(femur_muscles, colors):

    segments = muscle['line']

    projected_fibers = projected_segments(segments, [u1, u2])
    
    for fiber_segment, projected_fiber in zip(segments, projected_fibers):
        # plot fiber
        ax.plot(*fiber_segment.T, color=c)
        ax.plot(*projected_fiber.T, color=c, linestyle='--')

# plot v1 and v2
ax.plot(*np.array([np.zeros(3), u1]).T, color='g')
ax.plot(*np.array([np.zeros(3), u2]).T, color='r')


# plot plane

# get max coordinates from the fibers
max_x = int(max([fiber[:,0].max() for fiber in projected_fibers]))
max_y = int(max([fiber[:,1].max() for fiber in projected_fibers]))

xx, yy = np.meshgrid(range(2*max_x), range(2*max_y))

# vertical x-z plane
a,b,c,d, plot_plane = plane_abcd(v1, v2)

ax.plot_surface(xx, yy, plot_plane(xx, yy), alpha=0.5)

plt.show()
