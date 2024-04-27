from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt

from skspatial.objects import Point, Points, Plane
from skspatial.plotting import plot_3d
import map_muscles.muscle_template.xray_utils as xu
from scipy.spatial.distance import cdist

def generate_segment_points(A:Point, B:Point, distance=1.0):
    """
    Generates a Points along the line segment between points A and B.

    Args:
        A (Point): The starting point of the line segment.
        B (Point): The ending point of the line segment.
        distance (float): The distance between each generated point. Default is 1.0.

    Returns:
        Points: An instance of the Points class containing the generated points.
    """

    vector = B - A

    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude

    n_points = int(magnitude / distance)

    points = [A + unit_vector * i * distance for i in range(n_points)]
    
    return Points(points)




class Segment():
    def __init__(self, two_points):
        self.A = Point(two_points[0])
        self.B = Point(two_points[1])

    def points_plotter(self, **kwargs):
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
        plotter = Points([self.A, self.B]).plotter(**kwargs)

        return plotter
    
    def project_on_plane(self, plane: Plane):
        projected_A = plane.project_point(self.A)
        projected_B = plane.project_point(self.B)
        return Segment([projected_A, projected_B])
    
    def plot_segment(self, ax, **kwargs):
        ax.plot([self.A[0], self.B[0]], [self.A[1], self.B[1]], [self.A[2], self.B[2]], **kwargs)

    def get_points(self):
        return self.A, self.B
    
    def generate_segment_points(self, distance=1.0):
        """
        Generates a Points along the line segment between points A and B.

        Args:
            distance (float): The distance between each generated point. Default is 1.0.

        Returns:
            Points: An instance of the Points class containing the generated points.
        """

        return generate_segment_points(self.A, self.B, distance=distance)


        
    
class Fiber(Segment):
    pass

def get_segments_min_x(segments):
    return min([min(segment.A[0], segment.B[0]) for segment in segments])

def get_segments_max_x(segments):
    return max([max(segment.A[0], segment.B[0]) for segment in segments])

def get_segments_min_y(segments):
    return min([min(segment.A[1], segment.B[1]) for segment in segments])

def get_segments_max_y(segments):
    return max([max(segment.A[1], segment.B[1]) for segment in segments])

def get_segments_min_z(segments):
    return min([min(segment.A[2], segment.B[2]) for segment in segments])

def get_segments_max_z(segments):
    return max([max(segment.A[2], segment.B[2]) for segment in segments])

def get_segments_lims_x(segments):
    return [get_segments_min_x(segments), get_segments_max_x(segments)]

def get_segments_lims_y(segments):
    return [get_segments_min_y(segments), get_segments_max_y(segments)]

def get_segments_lims_z(segments):
    return [get_segments_min_z(segments), get_segments_max_z(segments)]

def get_unique_points(segments):
    points = np.array([segment.get_points() for segment in segments]).reshape(-1, 3)
    return Points(points).unique()

def get_unique_points_from_set_of_points(points):
    points = np.array(points).reshape(-1, 3)
    return Points(points).unique()


class Segments():
    segments: np.array # of Segment

    def __init__(self, segments):
        """
        Args:
            fibers (list): A list of Segment.
        """
        self.segments = np.array(segments)

    @classmethod
    def segments_from_points(cls, segments):
        """_summary_

        Args:
            segments (list): A list of two points tuple
        """
        return cls([Segment(segment) for segment in segments])

    def get_points(self):
        points = get_unique_points(self.segments)
        return points

    def generate_segment_points(self, distance=1.0):
        """
        Generates a Points along the line segments.

        Args:
            distance (float): The distance between each generated point. Default is 1.0.

        Returns:
            Points: An instance of the Points class containing the generated points.
        """
        points = []
        for segment in self.segments:
            points.extend(segment.generate_segment_points(distance=distance))
        
        return get_unique_points_from_set_of_points(points)


    def get_min_distance(self):
        """
        Get the minimum distance between all the points in the fibers.

        Returns:
            float: The minimum distance between all the points in the fibers.
        """
        points = self.get_points()

        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        return np.min(distances)

    def generate_all_linked_segment_points(self, distance=1.0):
        min_distance = self.get_min_distance()
        assert distance <= min_distance, \
            f"distance must be greater than min distance between points of the fibers; \n \
                min distance = {min_distance}."

        points = self.get_points()
        
        def unique_pairs(n):
            indices = np.arange(n)
            return np.array(list(zip(*np.triu_indices(n, k=1))))
        
        pairs = unique_pairs(len(points))

        linked_points = []
        for pair in pairs:
            A = Point(points[pair[0]])
            B = Point(points[pair[1]])
            segment_points = generate_segment_points(A, B, distance=distance)
            linked_points.extend(segment_points)

        return Points(linked_points).unique()

            
            

            
        



    def points_plotters(self, **kwargs):
        """
        Returns a list of plotters for each fiber.

        Args:
            **kwargs: Additional keyword arguments to be passed to the fiber plotter.

        Returns:
            list: A list of plotters for each fiber.
        """
        
        points = self.get_points()
        plotters = points.plotter(**kwargs)

        return plotters
        
    def plot_segments(self, ax, **kwargs):
        for segment in self.segments:
            segment.plot_segment(ax, **kwargs)

    def project_on_plane(self, plane: Plane):

        segments = [segment.project_on_plane(plane) for segment in self.segments]
        
        return Segments(segments)

    def get_min_x(self):
        return get_segments_min_x(self.segments)
    def get_max_x(self):
        return get_segments_max_x(self.segments)
    def get_min_y(self):
        return get_segments_min_y(self.segments)
    def get_max_y(self):
        return get_segments_max_y(self.segments)    
    def get_min_z(self):
        return get_segments_min_z(self.segments)
    def get_max_z(self):
        return get_segments_max_z(self.segments)
    def get_lims_x(self):
        return get_segments_lims_x(self.segments)
    def get_lims_y(self):
        return get_segments_lims_y(self.segments)
    def get_lims_z(self):
        return get_segments_lims_z(self.segments)
    def get_lims(self):
        return self.get_lims_x(), self.get_lims_y(), self.get_lims_z()

class Fibers(Segments):
    pass

def segments_to_points(segment: Segment, n:int=3):
    """
    Convert a line (represented by two points) to a set of n evenly spaced points 
    between the two points.
    """

    p1, p2 = segment.A, segment.B
    x = np.linspace(p1[0], p2[0], n)
    y = np.linspace(p1[1], p2[1], n)
    z = np.linspace(p1[2], p2[2], n)

    return x,y,z

def divide_segment(segment, n):
    """
    Divide a segment into n_segments segments (Segment) of equal length.

    Args:
        segment (Segment): A segment to be divided.
        n_segments (int): The number of segments to divide the segment into.

    Returns:
        list: np.array of n_segments segments.
    """

    segments = []

    x,y,z = segments_to_points(segment, n=n+1)

    for i in range(n):
        segment = np.array(
            [
            [x[i], y[i], z[i]],
            [x[i+1], y[i+1], z[i+1]],
            ]
        )

        segments.append(Segment(segment))
    # to array
    segments = np.array(segments)
    
    return segments

class SegmentedFiber(Segments):
    n_segments: int # number of segments

    def __init__(self, segment:Segment, n_segments:int):

        """
        Args:
            segments (list): A list of Segment
            n_segments (int): The number of segments to divide the fiber into.
        """
        Segments.__init__(self, divide_segment(segment, n_segments))
        self.n_segments = n_segments

class SegmentedFibers():
    n_segments: int
    segmented_fibers: list # of SegmentedFiber

    def __init__(self, segmented_fibers):
        """
        Args:
            segmented_fibers (list): A list of SegmentedFiber
        """
        self.segmented_fibers = segmented_fibers

        assert all([segmented_fiber.n_segments == segmented_fibers[0].n_segments for segmented_fiber in segmented_fibers]), \
            "SegmentedFibers.__init__(): All segmented fibers must have the same number of segments."

        self.n_segments = segmented_fibers[0].n_segments

    @classmethod
    def from_fibers(cls, fibers:Fibers, n_segments: int):
        """
        Args:
            fibers (Fibers): A Fibers object.
            n_segments (int): The number of segments to divide each fiber into.
        """
        segmented_fibers = [SegmentedFiber(segment, n_segments) for segment in fibers.segments]
        return cls(segmented_fibers)
    
    def get_points(self):
        points = [segmented_fiber.get_points() for segmented_fiber in self.segmented_fibers]
        return get_unique_points_from_set_of_points(points)

    def points_plotters(self, **kwargs):
        """
        Returns a list of plotters for each segmented fiber.

        Args:
            **kwargs: Additional keyword arguments to be passed to the segmented fiber plotter.

        Returns:
            list: A list of plotters for each segmented fiber.
        """
        plotters = []
        for segmented_fiber in self.segmented_fibers:
            plotters.append(segmented_fiber.points_plotters(**kwargs))
        
        return np.array(plotters)
    
    def plot_segments(self, ax, **kwargs):
        for segmented_fiber in self.segmented_fibers:
            segmented_fiber.plot_segments(ax, **kwargs)

    def project_on_plane(self, plane: Plane):
        segmented_fibers = [segmented_fiber.project_on_plane(plane) for segmented_fiber in self.segmented_fibers]
        return SegmentedFibers(segmented_fibers)
    
    def get_min_x(self):
        return min([segmented_fiber.get_min_x() for segmented_fiber in self.segmented_fibers])
    
    def get_max_x(self):
        return max([segmented_fiber.get_max_x() for segmented_fiber in self.segmented_fibers])
    
    def get_min_y(self):
        return min([segmented_fiber.get_min_y() for segmented_fiber in self.segmented_fibers])
    
    def get_max_y(self):
        return max([segmented_fiber.get_max_y() for segmented_fiber in self.segmented_fibers])
    
    def get_min_z(self):
        return min([segmented_fiber.get_min_z() for segmented_fiber in self.segmented_fibers])
    
    def get_max_z(self):
        return max([segmented_fiber.get_max_z() for segmented_fiber in self.segmented_fibers])
    
    def get_lims_x(self):
        return [self.get_min_x(), self.get_max_x()]
    
    def get_lims_y(self):
        return [self.get_min_y(), self.get_max_y()]
    
    def get_lims_z(self):
        return [self.get_min_z(), self.get_max_z()]
    
    def get_lims(self):
        return self.get_lims_x(), self.get_lims_y(), self.get_lims_z()
    

def surface_from_fibers(twoFibers, distance_between_points=1.):
    """
    Create a surface from two fibers.
    That is equally spaced points between the two fibers.

    Args:
        twoFibers (list): A list of two Fibers objects.
        distance_between_points (float, optional): The distance between points in the surface. Defaults to 1.

    Returns:
        Points: A Points object representing the surface.
    """
    fiberA, fiberB = twoFibers


class FiberSurface():

    surface: Points

    def __init__(self, twoFibers, n_along_fibers=10, n_across_fibers=10):
        self.surface = surface_from_fibers(twoFibers, n_along_fibers=n_along_fibers, n_across_fibers=n_across_fibers)

    def points_plotter(self, **kwargs):
        """
        Plotter method for the muscle template.
        It has to be unpacked in plot_3d() to be used: 
        
        e.g.:
        plot_3d(*FiberSurface.plotter())

        Args:
            **kwargs: Additional keyword arguments to be passed to the underlying plotter methods.

        Returns:
            tuple: A tuple containing the plots generated by the underlying plotter methods for muscle A and muscle B.
        """
        return self.surface.plotter(**kwargs)


class SurfacedFibers():
    n_along_fibers: int
    n_across_fibers: int
    fibers: list # of fiber
    surface: FiberSurface

    def __init__(self, fibers, n_along_fibers=10, n_across_fibers=10):
        ...


class Muscles():
    muscles: list # of Fibers

    def __init__(self, muscles):
        """
        Args:
            muscles (list): A list of Fibers
        """
        self.muscles = muscles

    @classmethod
    def muscles_from_df(cls, muscles_df, line_key='line'):
        return cls([Fibers.segments_from_points(muscle['line'].to_numpy()) for muscle in muscles_df])
    
    def get_points(self):
        points = []
        for muscle in self.muscles:
            points.extend(muscle.get_points())
        return get_unique_points_from_set_of_points(points)
    
    def points_plotter(self, **kwargs):
        """
        Returns a plotter for the muscles, representing the endpoints of the fibers.

        Args:
            **kwargs: Additional keyword arguments to be passed to the muscle plotter.

        Returns:
            list: A list of plotters for each muscle.
        """

        points = self.get_points()
        plotter = points.plotter(**kwargs)

        return plotter
    
    def plot_segments(self, ax, **kwargs):
        for muscle in self.muscles:
            muscle.plot_segments(ax, **kwargs)

    def generate_segment_points(self, distance=1.0):
        """
        Generates a Points along the line segments.

        Args:
            distance (float): The distance between each generated point. Default is 1.0.

        Returns:
            Points: An instance of the Points class containing the generated points.
        """
        points = []
        for muscle in self.muscles:
            points.extend(muscle.generate_segment_points(distance=distance))
        
        return get_unique_points_from_set_of_points(points)

def one_muscle_to_fibers(muscle, line_key='line'):
    """
    Convert a muscle df to a Fibers object.

    Parameters:
    muscle (dict): A dictionary representing a muscle.
    line_key (str, optional): The key in the muscle dictionary that contains the fiber data. Defaults to 'line'.

    Returns:
    Fibers: A Fibers object representing the muscle's fibers.
    """
    return Fibers.segments_from_points(muscle[line_key].to_numpy())

def muscles_to_fibers(muscles, line_key='line'):
    """
    Convert a list of muscles dfs to a list of Fibers.

    Parameters:
    muscles (list): A list of dictionaries representing muscles.
    line_key (str, optional): The key in each muscle dictionary that contains the fiber data. Defaults to 'line'.

    Returns:
    list: A list of Fibers objects, each representing a muscle's fibers.
    """
    return [Fibers.segments_from_points(muscle[line_key].to_numpy()) for muscle in muscles]


if __name__ == "__main__":

    muscles = xu.get_femur_muscles(remove=True)
    muscle = muscles[0]
    fibers = one_muscle_to_fibers(muscle)
    type(fibers)
    type(fibers.segments)
    fiber = fibers.segments[0]
    n = 4 # number of segments
    segmented_fiber = SegmentedFiber(fiber, n=n)
    type(segmented_fiber)
    type(segmented_fiber.segments)
    type(segmented_fiber.segments[0])
    
    fig, ax = plot_3d(
        segmented_fiber.points_plotters()
    )
    segmented_fiber.plot_segments(ax, c='k')

    plt.show()