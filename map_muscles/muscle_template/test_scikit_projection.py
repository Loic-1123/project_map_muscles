from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt

from skspatial.plotting import plot_3d
from skspatial.objects import Plane

import map_muscles.muscle_template.scikit_spatial_projection as ssp
import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf

np.random.seed(0)

def test_femur_muscles_plotting():

    muscles = xu.get_femur_muscles(remove=True)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    muscles_fibers = ssp.muscles_to_fibers(muscles)

    colors = vf.get_random_color_map(muscles)

    for muscle_fibers, c in zip(muscles_fibers, colors):
        muscle_fibers.plot_fibers(ax, c=c)

    plt.show()
    plt.close()

def test_projection_of_one_muscle(index=0):

    muscles = xu.get_femur_muscles(remove=True)
    muscle = muscles[index]
    fibers = ssp.one_muscle_to_fibers(muscle)

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]

    plane = Plane.from_points(points[0], points[1], points[2])

    projected_fibers = fibers.project_on_plane(plane)

    fig, ax = plot_3d(
        plane.plotter(alpha=0.5, lims_x=projected_fibers.get_lims_x(), lims_y=projected_fibers.get_lims_y()),
    )

    fibers.plot_fibers(ax, c='b')
    projected_fibers.plot_fibers(ax, c='r')

    plt.show()







if __name__ == "__main__":
    #test_femur_muscles_plotting()
    #test_projection_of_one_muscle()
    

