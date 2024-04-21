from _root_path import add_root, get_root_path
add_root()

from pathlib import Path

import numpy as np
import numpy.linalg as linalg
import scipy.spatial as spatial
import open3d as o3d
import matplotlib.pyplot as plt
import tqdm

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.fibers_object as fo

# Tait-Bryan angles convention: https://en.wikipedia.org/wiki/Euler_angles#Conventions
# z-y'-x'' (intrinsic rotations)
# yaw-pitch-roll
# first rotation around z, then y, then x
# yaw angle with respect to x axis (rotation around z)
# pitch angle with respect to x axis (rotation around z)

def compute_yaw(vec: np.ndarray) -> float:
    yaw = np.arctan2(vec[1], vec[0])

    return yaw

def compute_pitch(vec: np.ndarray) -> float:
    pitch = np.arctan2(vec[2], vec[0])

    return pitch

class Muscle():
    points: np.ndarray # 3D points representing the muscle surface, shape: (n, 3)

    name: str # Name of the muscle

    yaw: float # Yaw of the muscle to absolute coordinates

    pitch: float # Pitch of the muscle to absolute coordinates

    roll: float # Roll of the muscle around the muscle axis

    axis_points: np.ndarray # Axis of the muscle, represented by two points, shape: (2, 3)

    axis_vector: np.ndarray # Vector representing the axis of the muscle, shape: (3,)

    def __init__(self, points: np.ndarray, name:str,  axis_points:np.ndarray=None, roll:float=None):
        self.points = points
        self.name = name

        if np.any(axis_points, None):
            self.set_axis_points(axis_points, compute_dependend_attributes=True)

        else:
            self.axis_points = None
            self.axis_vector = None
            self.yaw = None
            self.pitch = None

        self.roll = roll

    @classmethod
    def from_array_file(cls, file_path: Path, name=None):

        points = np.load(file_path)

        if name is None:
            name = file_path.stem 
            
        return cls(points, name)    

    def compute_axis_vector(self) -> np.ndarray:
        axis = (self.axis_points[1] - self.axis_points[0])
        axis = axis / linalg.norm(axis)
        return axis
    
    def compute_self_yaw(self) -> float:
        return compute_yaw(self.axis_vector)
        
    def compute_self_pitch(self) -> float:
        return compute_pitch(self.axis_vector)
    
    def compute_set_yaw_pitch(self):
        self.yaw = self.compute_self_yaw()
        self.pitch = self.compute_self_pitch()
  
    def set_axis_points(self, axis_points: np.ndarray, compute_dependend_attributes=True):
        self.axis_points = axis_points

        if compute_dependend_attributes:
            self.axis_vector = self.compute_axis_vector()
            self.compute_set_yaw_pitch()

    def rotate(self, axis: np.ndarray, theta: float):

        assert self.axis_vector is not None, "Axis vector must be set before rotating."

        assert np.isclose(linalg.norm(axis), 1), "Axis must be a unit vector."

        rotation = spatial.transform.Rotation.from_rotvec(axis * theta)

        rotated_points = rotation.apply(self.points)

        new_axis_points = rotation.apply(self.axis_points)

        return Muscle(rotated_points, self.name, new_axis_points)  
    
    def roll_points(self, theta:float):

        assert self.roll is not None, "Roll must be set before rolling."

        axis = self.get_axis_vector()

        rotation = spatial.transform.Rotation.from_rotvec(axis * theta)

        rotated_points = rotation.apply(self.points)

        return Muscle(rotated_points, name=self.name, axis_points=self.axis_points, roll=self.roll + theta)
        
    def plot_axis_points(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.axis_points[:,0], self.axis_points[:,1], self.axis_points[:,2], **kwargs)
        return ax
    
    def plot_axis(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.quiver(self.axis_points[0,0], self.axis_points[0,1], self.axis_points[0,2], self.axis_vector[0], self.axis_vector[1], self.axis_vector[2], **kwargs)
        ax.plot([self.axis_points[0,0], self.axis_points[1,0]], [self.axis_points[0,1], self.axis_points[1,1]], [self.axis_points[0,2], self.axis_points[1,2]], **kwargs)
        return ax
    
    def plot_points(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], **kwargs)
        return ax
    
    def default_plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        # plot points little bit transparent
        ax = self.plot_points(ax, s=1, alpha=0.5, color='k')
        ax = self.plot_axis_points(ax, color='r')
        ax = self.plot_axis(ax, color='r')
        return ax
    


    
    

    
    



