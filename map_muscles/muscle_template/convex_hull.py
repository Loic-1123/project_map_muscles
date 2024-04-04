from _root_path import add_root, get_root_path
add_root()

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.segment_muscle as sm
import map_muscles.muscle_template.projection_on_plane as pp


def project_muscle_segment_fiber_divided(muscle_segment, U):
    projected_segments = []
    for section in muscle_segment:
        projected = pp.projected_segments(section, U)
        projected_segments.append(projected)
    projected_segments = np.array(projected_segments)
        
    return projected_segments

def projected_muscle_segment_points(muscle_segment, U, remove_z=True):
    projected_segments = project_muscle_segment_fiber_divided(muscle_segment, U)
    projected_points = projected_segments.reshape(-1, 3)

    if remove_z:
        projected_points = projected_points[:, :2]
    return projected_points

np.random.seed(0)
trfe = xu.get_femur_muscles()[-1]
fibers = trfe['line'].to_numpy()

nb = 4
segmented_muscle = sm.segment_muscle_by_dividing_fibers(fibers, n=nb)
segmented_muscle.shape

v1 = np.array([1.0,.0,.0])
v2 = np.array([.0,1.0,.0])

u1,u2 = pp.orthonormal_vectors(v1, v2)
plane_vectors = np.array([u1, u2])
a,b,c,d,plot_plane = pp.plane_abcd(u1, u2)

muscle_segment = segmented_muscle[0]
muscle_segment.shape
#(2,7,2,3)

projected_segments = project_muscle_segment_fiber_divided(muscle_segment, plane_vectors)
projected_segments.shape
#(2,7,2,3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['k', 'b']

for i, section in enumerate(projected_segments):
    for fiber in section:
        ax.plot(*fiber.T, color=colors[i])

# reduce to 2D
points = projected_muscle_segment_points(muscle_segment, plane_vectors)
hull = ConvexHull(points, qhull_options='QJ')
hull_points = points[hull.vertices]

# plot hull points
ax.plot(hull_points[:,0], hull_points[:,1], 'ro')

# plot hull
for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], 'r--')


plt.show()