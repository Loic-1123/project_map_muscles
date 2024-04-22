from _root_path import add_root, get_root_path
add_root()

from pathlib import Path

import numpy as np
import numpy.linalg as linalg
import scipy.spatial as spatial
import open3d as o3d
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

    pcd: o3d.geometry.PointCloud # Open3D point cloud object for visualization

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
        self.pcd = None

    @classmethod
    def from_array_file(cls, file_path: Path, name=None):
        """
        Create a new instance of the class from an .npy array file.

        Args:
            file_path (Path): The path to the array file representing the points.
            name (str, optional): The name of the instance. If not provided, the name will be set to the stem of the file path.

        Returns:
            cls: A new instance of the class.

        """
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

    def get_axis_centre(self) -> np.ndarray:
        """
        Calculate the center point of the axis.

        Returns:
            np.ndarray: The center point of the axis.
        """
        assert self.axis_points is not None,\
              "Axis points must be set before getting the center."
        return np.mean(self.axis_points, axis=0)

    def translate(self, translation: np.ndarray, new_name=None):
        translated_points = self.points + translation

        if new_name:
            name = new_name 
        else:
            name = self.name + "_translated"

        return Muscle(translated_points, name=name, axis_points=self.axis_points, roll=self.roll)



    def rotate(self, rotvec: np.ndarray, theta: float, new_name=None):
        """
        Rotate the muscle around the centre of its axis.

        Parameters:
            rotvec (np.ndarray): The rotation axis vector. Must be a unit vector.
            theta (float): The angle of rotation in radians.
            new_name (str, optional): The name of the rotated muscle. If not provided, the original name with "_rotated" appended will be used.

        Returns:
            Muscle: A new Muscle object representing the rotated muscle.

        Raises:
            AssertionError: If the muscle axis vector is not set or if the rotational axis is not a unit vector.
        """
        assert self.axis_vector is not None, "Muscle axis vector must be set before rotating."
        assert np.isclose(linalg.norm(rotvec), 1), "Rotational axis must be a unit vector."

        current_centre = self.get_axis_centre()

        # translation to origin
        translated_points = self.points - current_centre
        # rotation
        rotation = spatial.transform.Rotation.from_rotvec(rotvec * theta)
        rotated_points = rotation.apply(translated_points)
        # translation back
        rotated_points = rotated_points + current_centre

        # same with axis points
        axis_points = self.axis_points - current_centre
        rotated_axis_points = rotation.apply(axis_points)
        rotated_axis_points = rotated_axis_points + current_centre
        
        #TODO: compute roll that occured during rotation
        roll=None
        
        if new_name:
            name = new_name
        else:
            name = self.name + "_rotated"

        return Muscle(rotated_points, name=name, axis_points=rotated_axis_points, roll=roll)
    
    def roll_points(self, theta:float):
        #TODO
        pass

        
    
    def init_pcd(self):
        if self.pcd is not None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd = pcd

    def paint_uniform_color(self, color: np.ndarray):
        """
        Sets the color of the point cloud data (PCD).

        Parameters:
        color (np.ndarray): (r,g,b), the color to be applied to the PCD,
        values ranging between 0 and 1.

        Returns:
        None
        """
        self.init_pcd()
        self.pcd.paint_uniform_color(color)

    def draw_points(self, vis: o3d.visualization.Visualizer, color: np.ndarray = np.array([0,0,0])):
        self.init_pcd()
        self.pcd.paint_uniform_color(color)
        vis.add_geometry(self.pcd)

    def draw_axis(self, vis: o3d.visualization.Visualizer, length=1000, color: np.ndarray = np.array([1,0,0])):
        assert self.axis_points is not None, "Axis points must be set before drawing axis."

        axis = o3d.geometry.LineSet()

        axis_points = [self.axis_points[0]-0.5*length*self.axis_vector, self.axis_points[0] + 0.5*length*self.axis_vector]

        axis.points = o3d.utility.Vector3dVector(self.axis_points)
        axis.lines = o3d.utility.Vector2iVector([[0,1]])
        axis.paint_uniform_color(color)
        
        vis.add_geometry(axis)

    def draw_default(self, vis: o3d.visualization.Visualizer):
        self.draw_points(vis)
        self.draw_axis(vis)
    

    


    
    

    
    



