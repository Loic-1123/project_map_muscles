from _root_path import add_root
add_root()

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.xray_utils as xu



def line_to_points(line, n=3):
    """
    Convert a line (represented by two points) to a set of n evenly spaced points 
    between the two points.
    """

    p1, p2 = line
    x = np.linspace(p1[0], p2[0], n)
    y = np.linspace(p1[1], p2[1], n)
    z = np.linspace(p1[2], p2[2], n)

    return x,y,z

def divide_fiber(fiber, n=4):
    """
    Divide a fiber into n segments
    """
    segments = []

    x,y,z = line_to_points(fiber, n=n+1)

    for i in range(n):
        segment = np.array(
            [
            [x[i], y[i], z[i]],
            [x[i+1], y[i+1], z[i+1]],
            ]
        )
        segments.append(segment)
    # to array
    segments = np.array(segments)
    
    return segments

def divided_fibers(fibers, n=4):
    divided_fibers = []
    for fiber in fibers:
        segments = divide_fiber(fiber, n=n)
        divided_fibers.append(segments)

    # to array
    divided_fibers = np.array(divided_fibers)
    return divided_fibers

def plot_segment(ax, segment, color='b'):
    p1, p2 = segment
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)

def plot_point(ax, p, color='r'):
    ax.scatter(p[0], p[1], p[2], color=color)

def plot_points(ax, points, color='r'):
    for p in points:
        plot_point(ax, p, color=color)

def plot_segments(ax, segmented_fiber, fiber_color='b', show_points=True, point_color='r'):
    for segment in segmented_fiber:
        plot_segment(ax, segment, color=fiber_color)
        if show_points:
            p1, p2 = segment
            plot_points(ax, [p1, p2], color=point_color)

def check_fibers_same_nb_segments(segmented_fibers):
    segments_shapes = np.array([fiber.shape[0] for fiber in segmented_fibers])
    assert np.all(segments_shapes == segments_shapes[0]), \
        "Fibers do not have the same number of segments"
    
def segment_muscle(muscle_fibers, n=3):
    segmented_fibers = divided_fibers(muscle_fibers, n=n+1)
    check_fibers_same_nb_segments(segmented_fibers)

    segmented_muscle = []

    for i in range(n):
        segments = [segmented_fibers[:,i], segmented_fibers[:,i+1]]
        segmented_muscle.append(segments)

    # to array
    segmented_muscle = np.array(segmented_muscle)
    return segmented_muscle

def plot_segmented_muscle(ax, segmented_muscle, segmentation_colors, show_points=True):
    assert len(segmentation_colors) == segmented_muscle.shape[0], \
        "Number of colors must match the number of muscle segments"
    
    for i, muscle_segment in enumerate(segmented_muscle):
        for fiber_segment in muscle_segment:
            plot_segments(ax, fiber_segment, fiber_color=segmentation_colors[i], show_points=show_points)


muscles = xu.get_femur_muscles()
trfe = muscles[-1]

trfe.describe()
# fuse the sets of points A and B
fibers = trfe['line'].to_numpy()
fibers.shape

nb= 4
segmented_muscle = segment_muscle(fibers, n=nb)
segmented_muscle.shape
""""
 (4, 2, 7, 2, 3): 
 4 muscle segments, 
 2 fiber segments per muscle segment, 
 7 fibers, 2 points per fiber, 3 coordinates per point
"""

