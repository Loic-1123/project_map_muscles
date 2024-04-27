from _root_path import add_root, get_root_path
add_root()

from pathlib import Path

import numpy as np
np.random.seed(0)
import numpy.linalg as linalg
import scipy.spatial as spatial
import open3d as o3d
import matplotlib.pyplot as plt

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.fibers_object as fo


# Tait-Bryan angles convention: https://en.wikipedia.org/wiki/Euler_angles#Conventions
# z-y'-x'' (intrinsic rotations)
# yaw-pitch-roll
# first rotation around z, then y, then x
# yaw angle with respect to x axis (rotation around z); 
# pitch angle with respect to x axis (rotation around z)

def compute_yaw(vec: np.ndarray) -> float:
    """
    Compute the yaw angle (in radians) given a 3D vector.

    Parameters:
    vec (np.ndarray): A 3D vector represented as a NumPy array.

    Returns:
    float: The yaw angle in radians.

    """
    yaw = np.arctan2(vec[1], vec[0])

    return yaw

def compute_pitch(vec: np.ndarray) -> float:
    """
    Compute the pitch angle of a vector in 3D space.

    Parameters:
    vec (np.ndarray): The input vector in the form of a numpy array with shape (3,).

    Returns:
    float: The pitch angle in radians.

    """
    pitch = np.arctan2(vec[2], vec[0])

    return pitch

def compute_vector_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    vec = p2 - p1
    vec = vec / linalg.norm(vec)
    return vec

def generate_line(center: np.ndarray, vector: np.ndarray, length: float) -> o3d.geometry.LineSet:
    """
    Generate a line segment in 3D space: a o3d.geometry.Lineset() object.

    Parameters:
        center (np.ndarray): The center point of the line segment. Must be a 3D point.
        vector (np.ndarray): The direction vector of the line segment. Must be a 3D vector.
        length (float): The length of the line segment.

    Returns:
        o3d.geometry.LineSet: A LineSet object representing the line segment.

    Raises:
        AssertionError: If the center or vector shapes are not (3,).
    """

    assert center.shape == (3,), "Center must be a 3D point."
    assert vector.shape == (3,), "Vector must be a 3D vector."

    start = center - 0.5 * length * vector
    end = center + 0.5 * length * vector

    points = o3d.utility.Vector3dVector([start, end])
    lines_idx = o3d.utility.Vector2iVector([[0, 1]])

    return o3d.geometry.LineSet(points, lines_idx)

def get_equally_spaced_colors(n:int, cmap='hsv', rm_alpha_channel=True):
    """
    Generate n equally spaced colors.

    Parameters:
    - n (int): The number of colors to generate.

    Returns:
    - colors (np.ndarray): The array of colors.
    """

    cmap = plt.get_cmap(cmap)

    colors = np.array([cmap(i/n) for i in range(n)])

    # remove alpha channel
    if rm_alpha_channel:
        colors = colors[:, :-1]

    return colors

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
        return compute_vector_from_points(self.axis_points[0], self.axis_points[1])
    
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
            name = self.name

        return Muscle(translated_points, name=name, axis_points=self.axis_points, roll=self.roll)

    def rotate(self, rotvec: np.ndarray, theta: float, new_name=None):
        """
        Rotate the muscle around the centre of its axis.

        Calculation of the new roll is not implemented yet.

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
        
        if new_name:
            name = new_name
        else:
            name = self.name + "_rotated"

        return Muscle(rotated_points, name=name, axis_points=rotated_axis_points)
    
    def roll_points(self, theta:float):
        rotational_axis = self.axis_vector
        rotated_muscle = self.rotate(rotational_axis, theta)

        return rotated_muscle
        
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
        assert self.axis_points is not None, "Axis points must be et before drawing axis."

        center = self.get_axis_centre()

        axis = generate_line(center, self.axis_vector, length)
        axis.paint_uniform_color(color)
        
        vis.add_geometry(axis)

    def draw_default(self, vis: o3d.visualization.Visualizer):
        self.draw_points(vis)
        self.draw_axis(vis)

    def project_points_on_xy_plane(self, remove_z_axis=False):

        points = self.points[:, :2]
        
        if remove_z_axis:
            return points
        else:
            return np.hstack((points, np.zeros((points.shape[0], 1))))
    
    def get_map2d(self):
        return IndividualMap2d(
            self.project_points_on_xy_plane(remove_z_axis=True), 
            axis_points=self.axis_points,
            name=self.name
            )

    def centered_on_axis_point(self):
        return self.translate(-self.axis_points[0])
    
class MuscleMap():
    
    name: str # Name of the muscle map

    muscles: list # List of Muscle objects
    
    axis_points: np.ndarray # Axis points of the muscle map, shape: (n, 3)

    axis_vector: np.ndarray # Axis vector of the muscle map, shape: (3,)

    def __init__(self, muscles: list, axis_points: np.ndarray=None, name=None, roll=None, compute_dependend_attributes=True):
        self.muscles = muscles

        if np.any(axis_points, None):
            self.set_axis_points(axis_points, compute_dependend_attributes=compute_dependend_attributes)

        else:
            self.axis_points = None
            self.axis_vector = None

        if name is None:
            name = "MuscleMap"

        self.name = name

        self.roll = roll

    @classmethod
    def from_directory(cls, dir_path: Path, nb_muscles:int=5, axis_points: np.ndarray=None, name=None, roll=None):
        """
        Create a new instance of the class from a directory containing .npy files.

        Args:
            directory (Path): The path to the directory containing the .npy files.
            axis_points (np.ndarray, optional): The axis points of the muscle map. If not provided, the axis points will be set to the mean of the muscle axis points.
            name (str, optional): The name of the instance. If not provided, the name will be set to the stem of the directory path.
            roll (float, optional): The roll of the muscle map around the axis.

        Returns:
            cls: A new instance of the class.

        """

        files = list(dir_path.glob('*.npy'))

        assert len(files)==nb_muscles, f'Expected {nb_muscles} muscle files, got {len(files)}'

        muscles = [Muscle.from_array_file(file) for file in files[:nb_muscles]]

        if name is None:
            name = dir_path.stem

        return cls(muscles, axis_points=axis_points, name=name, roll=roll)

    def set_axis_points(self, axis_points: np.ndarray, compute_dependend_attributes=True):
        self.axis_points = axis_points

        if compute_dependend_attributes:
            self.axis_vector = self.compute_axis_vector()
            self.yaw = self.compute_self_yaw()
            self.pitch = self.compute_self_pitch()

            for muscle in self.muscles:
                muscle.set_axis_points(axis_points, compute_dependend_attributes=compute_dependend_attributes)
    
    def set_name(self, name: str):
        self.name = name
    
    def init_roll(self):
        self.roll = 0

    def compute_axis_vector(self) -> np.ndarray:
        return compute_vector_from_points(self.axis_points[0], self.axis_points[1])
    
    def compute_self_yaw(self) -> float:
        return compute_yaw(self.axis_vector)
    
    def compute_self_pitch(self) -> float:
        return compute_pitch(self.axis_vector)
    
    def compute_set_yaw_pitch(self):
        self.yaw = self.compute_self_yaw()
        self.pitch = self.compute_self_pitch()

    def get_axis_centre(self) -> np.ndarray:
        """
        Calculate the center point of the axis.

        Returns:
            np.ndarray: The center point of the axis.
        """
        assert self.axis_points is not None,\
              "Axis points must be set before getting the center."
        return np.mean(self.axis_points, axis=0)

    def get_axis_points(self) -> np.ndarray:
        return self.axis_points

    def get_com(self) -> np.ndarray:
        """
        Calculate the center of mass of the muscle map.

        Returns:
            np.ndarray: The center of mass of the muscle map.
        """
        
        points = []

        for muscle in self.muscles:
            points.extend(muscle.points)

        points = np.array(points)

        com = np.mean(points, axis=0)

        return com

    def translate(self, translation: np.ndarray, new_name=None):
        translated_muscles = [muscle.translate(translation) for muscle in self.muscles]
        translated_axis_points = self.axis_points + translation

        if new_name:
            name = new_name
        else:
            name = self.name

        return MuscleMap(translated_muscles, translated_axis_points, name=name)
    
    def rotate(self, rotvec: np.ndarray, theta: float, new_name=None):
        
        assert self.axis_points is not None, "Axis points must be set before rotating."

        rotated_muscles = [muscle.rotate(rotvec, theta) for muscle in self.muscles]

        new_axis_points = rotated_muscles[0].axis_points

        #TODO: compute roll that occured during rotation
        new_roll=None

        if new_name:
            name = new_name
        else:
            name = self.name + "_rotated"
        
        return MuscleMap(rotated_muscles, axis_points=new_axis_points, name=name)
    
    def to_yaw(self, yaw: float):
        """
        Rotate the muscle map to a given yaw angle.

        Parameters:
            yaw (float): The yaw angle in radians.

        Returns:
            MuscleMap: A new MuscleMap object representing the rotated muscle map.
        """
        rotvec = np.array([0, 0, 1])
        m = self.rotate(rotvec, yaw - self.yaw)
        m.roll = self.roll
        return m
    
    def to_pitch(self, pitch: float):
        """
        Rotate the muscle map to a given pitch angle.

        Parameters:
            pitch (float): The pitch angle in radians.

        Returns:
            MuscleMap: A new MuscleMap object representing the rotated muscle map.
        """
        rotvec = np.array([0, 1, 0])
        m = self.rotate(rotvec, self.pitch - pitch)
        m.roll = self.roll
        return m
    
    def roll_points(self, theta:float, set_new_roll=True):
        assert self.roll is not None, "Roll must be set before rolling."
        
        rotational_axis = self.axis_vector

        rotated_map = self.rotate(rotational_axis, theta)

        if set_new_roll:
            rotated_map.roll = self.roll + theta
        
        return rotated_map

    def draw_points(self, vis: o3d.visualization.Visualizer, colors: np.ndarray = None):

        if colors is None:
            colors = np.zeros((len(self.muscles), 3))

        for muscle, color in zip(self.muscles, colors):
            muscle.draw_points(vis, color)

    def draw_axis(self, vis: o3d.visualization.Visualizer, length=2000, color: np.ndarray = np.array([1,0,0])):
        
        assert self.axis_points is not None, "Axis points must be set before drawing axis."

        axis = generate_line(self.get_axis_centre(), self.axis_vector, length)
        axis.paint_uniform_color(color)
        
        vis.add_geometry(axis)

    def draw_axis_points(self, vis: o3d.visualization.Visualizer, radius=10.0, color: np.ndarray = np.array([1,0,0])):
        assert self.axis_points is not None, "Axis points must be set before drawing axis."

        for point in self.axis_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(point)
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere)


    def draw_default(self, vis: o3d.visualization.Visualizer):
        colors = get_equally_spaced_colors(len(self.muscles))
        self.draw_points(vis, colors=colors)
        self.draw_axis(vis, color=np.array([0,0,0]))
        self.draw_axis_points(vis, color=np.array([0,0,0]))
    
    def get_map2d(self):
        individual_maps = [muscle.get_map2d() for muscle in self.muscles]
        return Map2d(individual_maps, axis_points=self.axis_points[:, :2])
    
    def get_points(self):
        return np.vstack([muscle.points for muscle in self.muscles])
    
    def centered_on_axis_point(self):
        return self.translate(-self.axis_points[0])
    




class IndividualMap2d():
    name = str # Name of the map
    axis_points: np.ndarray # Axis points of the map, shape: (n, 2)
    points: np.ndarray # 2D points representing the map, shape: (n, 2)

    def __init__(self, points: np.ndarray, axis_points: np.ndarray=None, name=None):
        self.points = points

        if np.any(axis_points, None):
            self.set_axis_points(axis_points)
        else:
            self.axis_points = None

        self.name = name
    
    def set_axis_points(self, axis_points: np.ndarray):
        self.axis_points = axis_points

    def set_name(self, name: str):
        self.name = name

    def translate(self, translation_vec: np.ndarray):
        translated_points = self.points + translation_vec
        translated_axis_points = self.axis_points + translation_vec
        return IndividualMap2d(translated_points, axis_points=translated_axis_points, name=self.name)
    
    def get_points(self):
        return self.points
    
    def scale(self, scale_factor: float, warning=True): 
        if not np.allclose(self.axis_points[0], np.zeros(2)) and warning:
            print("Map: axis_points[0] not at origin, scaling might not be accurate.")

        scaled_points = self.points * scale_factor
        scaled_axis_points = self.axis_points * scale_factor
        return IndividualMap2d(scaled_points, axis_points=scaled_axis_points, name=self.name)

class Map2d():
    individual_maps: list # List of IndividualMap2d objects
    axis_points: np.ndarray # Axis points of the map, shape: (n, 2)

    def __init__(self, individual_maps: list, axis_points: np.ndarray=None):
        self.individual_maps = individual_maps

        assert axis_points.shape == (2, 2), "Axis points must be a 2x2 array."

        if np.any(axis_points, None):
            self.set_axis_points(axis_points)
        else:
            self.axis_points = None

    def set_axis_points(self, axis_points: np.ndarray):
        self.axis_points = axis_points
        for map in self.individual_maps:
            map.set_axis_points(axis_points)

    def get_axis_points(self):
        return self.axis_points
    
    def get_points(self):
        return np.vstack([map.points for map in self.individual_maps])
    
    def translate(self, translation_vec: np.ndarray):
        translated_maps = [mmap.translate(translation_vec) for mmap in self.individual_maps]
        translated_axis_points = self.axis_points + translation_vec
        return Map2d(translated_maps, axis_points=translated_axis_points)
    
    def scale(self, scale_factor: float, warning=True):
        scaled_maps = [mmap.scale(scale_factor, warning=warning) for mmap in self.individual_maps]
        scaled_axis_points = self.axis_points * scale_factor
        return Map2d(scaled_maps, axis_points=scaled_axis_points)
