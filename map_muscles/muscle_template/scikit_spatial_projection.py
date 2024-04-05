from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt

from skspatial.objects import Point, Points, Plane, LineSegment
from skspatial.transformation import transform_coordinates
from skspatial.plotting import plot_3d

from dataclasses import dataclass

import map_muscles.muscle_template.projection_on_plane as pp

class Segment():
    def __init__(self, two_points):
        self.A = Point(two_points[0])
        self.B = Point(two_points[1])

    def plotter(self, **kwargs):
        """
        Plotter method for the muscle template.
        It has to be unpacked in plot_3d() to be used: 
        
        e.g.:
        plot_3d(*Segment.plane_projection())

        Args:
            **kwargs: Additional keyword arguments to be passed to the underlying plotter methods.

        Returns:
            tuple: A tuple containing the plots generated by the underlying plotter methods for muscle A and muscle B.
        """
        plotter_A = self.A.plotter(**kwargs)
        plotter_B = self.B.plotter(**kwargs)

        return plotter_A, plotter_B
    
    def project_on_plane(self, plane: Plane):
        projected_A = plane.project_point(self.A)
        projected_B = plane.project_point(self.B)
        return Segment([projected_A, projected_B])
    
    def plot_segment(self, ax, **kwargs):
        ax.plot([self.A[0], self.B[0]], [self.A[1], self.B[1]], [self.A[2], self.B[2]], **kwargs)

@dataclass
class SegmentedFiber():
    segments: list

def fibers_to_segments(fibers):
    return [Segment(fiber) for fiber in fibers]


v1 = np.array([1.0,.0,.0])
v2 = np.array([.0,1.0,.0])

u1,u2 = pp.orthonormal_vectors(v1, v2)

points = Points([[0, 0, 0], v1, v2])

plane = Plane.from_points(points[0], points[1], points[2])
segment = Segment([[0, 0, 1], [1, 1, 3]])

projected_segment = segment.project_on_plane(plane)

fig, ax = plot_3d(
    plane.plotter(alpha=0.2),
    *segment.plotter(c='k'),
    *projected_segment.plotter(c='r'),
)

segment.plot_segment(ax, c='k')
projected_segment.plot_segment(ax, c='r')

plt.show()