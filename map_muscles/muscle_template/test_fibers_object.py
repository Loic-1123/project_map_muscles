from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt

from skspatial.plotting import plot_3d
from skspatial.objects import Plane

import map_muscles.muscle_template.fibers_object as ssp
import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf

np.random.seed(0)

muscles = xu.get_femur_muscles(remove=True)


def test_femur_muscles_plotting():
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    muscles_fibers = ssp.muscles_to_fibers(muscles)

    colors = vf.get_random_color_map(muscles)

    for muscle_fibers, c in zip(muscles_fibers, colors):
        muscle_fibers.plot_segments(ax, c=c)

    plt.show()
    plt.close()

def test_projection_of_one_muscle(index=0):
    
    muscle = muscles[index]
    fibers = ssp.one_muscle_to_fibers(muscle)

    points1 = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
    plane1 = Plane.from_points(points1[0], points1[1], points1[2])
    projected_fibers1 = fibers.project_on_plane(plane1)

    points2 = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
    plane2 = Plane.from_points(points2[0], points2[1], points2[2])
    projected_fibers2 = fibers.project_on_plane(plane2)

    points3 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    plane3 = Plane.from_points(points3[0], points3[1], points3[2])
    projected_fibers3 = fibers.project_on_plane(plane3)


    fig, ax = plot_3d(
        
        plane1.plotter(
            alpha=0.5, 
            lims_x=projected_fibers1.get_lims_x(), 
            lims_y=projected_fibers1.get_lims_y()
            ),

        plane2.plotter(
            alpha=0.5,

            lims_x=projected_fibers2.get_lims_x(), 
            lims_y=projected_fibers2.get_lims_y()
            ),

        plane3.plotter(
            alpha=0.5, 
            lims_x=projected_fibers3.get_lims_y(), 
            lims_y=projected_fibers3.get_lims_z()
            )
    )

    fibers.plot_segments(ax, c='k')
    projected_fibers1.plot_segments(ax, c='r')
    projected_fibers2.plot_segments(ax, c='b')
    projected_fibers3.plot_segments(ax, c='y')

    plt.show()

    plt.close()

def test_segmentation_on_one_fiber(index=0):
    muscle = muscles[index]
    fibers = ssp.one_muscle_to_fibers(muscle)
    fiber = fibers.segments[0]
    n = 4 # number of segments
    segmented_fiber = ssp.SegmentedFiber(fiber, n_segments=n)
    
    fig, ax = plot_3d(
        segmented_fiber.points_plotters(c='r')
    )
    segmented_fiber.plot_segments(ax, c='k')

    plt.show()
    plt.close()

def test_segmentation_SegmentedFibers(index=0):
    muscle = muscles[index]
    fibers = ssp.one_muscle_to_fibers(muscle)
    n = 4
    segmented_fibers = ssp.SegmentedFibers.from_fibers(fibers, n_segments=n)

    point_plotters = segmented_fibers.points_plotters(c='r', s=5)

    fig, ax = plot_3d(
        *point_plotters
    )

    segmented_fibers.plot_segments(ax, c='k')


    plt.show()
    plt.close()

def test_surface_between_two_fibers(n_along_fibers=100, n_across_fibers=10, index=0):
    muscle = muscles[index]
    fibers = ssp.one_muscle_to_fibers(muscle)
    fiber1 = fibers.segments[0]
    fiber2 = fibers.segments[1]

    surface = ssp.FiberSurface([fiber1, fiber2], n_along_fibers=n_along_fibers, n_across_fibers=n_across_fibers)

    fig, ax = plot_3d(
        fiber1.points_plotter(c='k'),
        fiber2.points_plotter(c='k'),
        surface.points_plotter(c='r', s=5),
    )

    fiber1.plot_segment(ax, c='k')
    fiber2.plot_segment(ax, c='k')
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    #test_femur_muscles_plotting()
    test_projection_of_one_muscle()
    #test_segmentation_on_one_fiber()
    #test_segmentation_SegmentedFibers()
    #test_surface_between_two_fibers(10e6, 10e5)

