from _root_path import add_root, get_root_path
add_root()

from pathlib import Path

import numpy as np
np.random.seed(0)
import numpy.linalg as linalg
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
import matplotlib.pyplot as plt

import map_muscles.muscle_template.xray_utils as xu
import map_muscles.muscle_template.visualize_leg_fibers as vf
import map_muscles.muscle_template.fibers_object as fo


# Using Euler angles: https://en.wikipedia.org/wiki/Euler_angles#Classic_Euler_angles

R, G, B = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
K, W = np.array([0,0,0]), np.array([1,1,1])

def assert_unit_vector(vector: np.ndarray, string=None):
    if not np.isclose(linalg.norm(vector), 1):
        raise AssertionError(f"Vector must be a unit vector. Got {vector}. {string}")
    
def assert_not_aligned_with_z(vector: np.ndarray, string=None):
    if np.allclose([vector[0], vector[1]],  0):
        raise AssertionError(f"Vector must not be aligned with the z-axis. {string}")

def compute_alpha(z_vector: np.ndarray, print_warning=True, raise_warning=False) -> float:
    """
    Compute the alpha angle of a vector in 3D space. 
    Assuming the vector was aligned with the z-axis before rotation.

    Parameters:
    z_vector (np.ndarray): The input vector in the form of a numpy array with shape (3,).

    Returns:
    float: The alpha angle in radians.

    """
    assert_unit_vector(z_vector, " In compute_alpha()")
    # formula from https://en.wikipedia.org/wiki/Euler_angles#Proper_Euler_angles
    z1, z2, z3 = z_vector[0], z_vector[1], z_vector[2]

    if np.isclose(np.abs(z3), 1, atol=1e-07):
        if z3 > 0:
            alpha = 0
        else:
            alpha = np.pi

        warning_str = "In compute_alpha(): If z3 is 1 or -1, beta=0 or pi, Alpha is confounded with Gamma. Returns alpha=0."
        if print_warning: print(warning_str)
        if raise_warning: raise Warning(warning_str)    
    else:
        equa = -z2/np.sqrt(1-z3**2)
        
        if np.isclose(equa, 1, atol=1e-07):
            alpha = 0
        elif np.isclose(equa, -1, atol=1e-07):
            equa = -1
            alpha = np.pi

        alpha = np.arctan2(z1, -z2)

    return alpha

def compute_beta(z_vector: np.ndarray) -> float:
    """
    Compute the beta angle of a vector in 3D space. 
    Assuming the vector was aligned with the z-axis before rotation.

    Parameters:
    z_vector (np.ndarray): The input vector in the form of a numpy array with shape (3,).

    Returns:
    float: The beta angle in radians.

    """
    assert_unit_vector(z_vector, " In compute_beta()")
    # formula from https://en.wikipedia.org/wiki/Euler_angles#Proper_Euler_angles
    beta = np.arccos(z_vector[2])

    return beta

def compute_gamma(x_vector:np.ndarray, y_vector:np.ndarray, print_warning=True, raise_warning=False) -> float:
    """
    Compute the gamma angle of a vector in 3D space. 
    Assuming the vectos were aligned with the x and y axes before rotation.

    Parameters:
    x_vector (np.ndarray): The input vector in the form of a numpy array with shape (3,). Rotated x-axis.
    y_vector (np.ndarray): The input vector in the form of a numpy array with shape (3,). Rotated y-axis.

    Returns:
    float: The gamma angle in radians.

    Raises:
    ValueError: If y3 is close to 0, indicating that alpha and gamma are confounded and beta is 0.

    """
    assert_unit_vector(x_vector, " In compute_gamma()")
    assert_unit_vector(y_vector, " In compute_gamma()")

    x3 =x_vector[2]
    y3 =y_vector[2]

    if np.isclose(y3, 0):
        warning_str = "In compute_gamma(): If y3 is 0,\
              Beta = 0, hence, alpha and gamma angles are confounded. Returns angle between absolute x-axis given x_vector."
        if raise_warning: raise Warning(warning_str)
        elif print_warning: print(warning_str)

        gamma = np.arctan2(x_vector[1], x_vector[0])

    else:
        # formula from https://en.wikipedia.org/wiki/Euler_angles#Proper_Euler_angles
        gamma = np.arctan2(x3, y3)
    return gamma

def compute_normal_vector_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
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

    axis_points: np.ndarray # Axis of the muscle, represented by two points, shape: (2, 3)

    alpha: float # Alpha angle of the muscle, in radians

    beta: float # Beta angle of the muscle, in radians

    gamma: float # Gamma angle of the muscle, in radians

    x_vector: np.ndarray # x-axis of the relative coordinate system of the muscle

    y_vector: np.ndarray # y-axis of the relative coordinate system of the muscle

    z_vector: np.ndarray # z-axis of the relative coordinate system of the muscle. 
    #Also axis vector. The muscle axis is assumed to be the z-axis of the relative coordinate system.

    pcd: o3d.geometry.PointCloud # Open3D point cloud object for visualization

    def __init__(self, points: np.ndarray, name:str=None,  axis_points:np.ndarray=None, gamma:float=0):
        self.points = points
        self.name = name
        self.pcd = None

        if np.any(axis_points, None):
            self.set_axis_points(axis_points, gamma=gamma, compute_dependend_attributes=True)

        else:
            self.axis_points = None
            self.set_xyz_vectors_to_None()
            self.set_angles_to_None()
            
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

        ref_point = self.axis_points[0]

        # translation to origin
        translated_points = self.points - ref_point
        
        # rotation
        rotation = R.from_rotvec(rotvec * theta)
        rotated_points = rotation.apply(translated_points)
        # translation back
        rotated_points = rotated_points + ref_point

        # same with axis points
        axis_points = self.axis_points - ref_point
        rotated_axis_points = rotation.apply(axis_points)
        rotated_axis_points = rotated_axis_points + ref_point
        
        if new_name:
            name = new_name
        else:
            name = self.name + "_rotated"

        return Muscle(rotated_points, name=name, axis_points=rotated_axis_points)

    def project_points_on_xy_plane(self, remove_z_axis=False):

        points = self.points[:, :2]
        
        if remove_z_axis:
            return points
        else:
            return np.hstack((points, np.zeros((points.shape[0], 1))))
    def centered_on_axis_point(self):
        return self.translate(-self.axis_points[0])
    
    def scale(self, scale_factor: float):
            """
            Scales the muscle by the given scale factor.
            Relative to the axis point: Position of the first axis point is preserved.

            Args:
                scale_factor (float): The factor by which to scale the muscle.

            Returns:
                Muscle: The scaled muscle.
            """

            centered_Muscle = self.centered_on_axis_point()

            scaled_points = centered_Muscle.get_points() * scale_factor
            scaled_axis_points = centered_Muscle.axis_points * scale_factor

            # translate back
            scaled_points = scaled_points + self.axis_points[0]
            scaled_axis_points = scaled_axis_points + self.axis_points[0]

            return Muscle(scaled_points, name=self.name, axis_points=scaled_axis_points, roll=self.roll)
    def generate_map2d(self):
        return IndividualMap2d(
            self.project_points_on_xy_plane(remove_z_axis=True), 
            axis_points=self.axis_points,
            name=self.name
            )
    
    # Getters
    def get_points(self):
        return self.points
    
    def get_name(self):
        return self.name
    
    def get_axis_points(self):
        return self.axis_points
    
    def get_x_vector(self):
        return self.x_vector
    
    def get_y_vector(self):
        return self.y_vector
    
    def get_z_vector(self):
        return self.z_vector
    
    def get_alpha(self):
        return self.alpha
    
    def get_beta(self):
        return self.beta
    
    def get_gamma(self):
        return self.gamma
    
    def get_pcd(self):
        assert self.pcd is not None, "Point cloud data not initialized."
        return self.pcd

    
    # Setters / Initializers
      
    def set_axis_points(self, axis_points: np.ndarray, gamma=0, compute_dependend_attributes=True):
        self.axis_points = axis_points

        if compute_dependend_attributes:
            self.init_z_vector()
            self.init_aplha()
            self.init_beta()
            self.init_default_gamma_and_x_y_vectors(gamma=gamma)

    def set_name(self, name: str):
        self.name = name

    def set_xyz_vectors_to_None(self):
        self.x_vector = None
        self.y_vector = None
        self.z_vector = None

    def set_angles_to_None(self):
        self.alpha = None
        self.beta = None
        self.gamma = None

    def init_z_vector(self):
        self.assert_axis_points("In init_z_vector()")
        self.z_vector= self.compute_z_vector()

    def init_aplha(self):
        self.assert_z_vector("In init_alpha()")
        self.alpha = compute_alpha(self.z_vector)

    def init_beta(self):
        self.assert_z_vector("In init_beta()")
        self.beta = compute_beta(self.z_vector)

    def init_gamma(self):
        self.assert_x_vector("In init_gamma()")
        self.assert_y_vector("In init_gamma()")
        self.gamma = compute_gamma(self.x_vector, self.y_vector)

    def init_default_gamma_and_x_y_vectors(self, gamma=0):
        self.assert_alpha("In init_default_gamma_and_x_y_vectors()")
        self.assert_beta("In init_default_gamma_and_x_y_vectors()")
        self.assert_z_vector("In init_default_gamma_and_x_y_vectors()")

        self.gamma = gamma

        r = Rot.from_euler('ZXZ', [self.alpha, self.beta, self.gamma])
        self.x_vector = r.apply(np.array([1,0,0]))
        self.y_vector = r.apply(np.array([0,1,0]))
                    
    def init_pcd(self):
        if self.pcd is not None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd = pcd


    # Computations

    def compute_z_vector(self) -> np.ndarray:
        z_vec = compute_normal_vector_from_points(self.axis_points[0], self.axis_points[1])
        return z_vec
    
    def compute_angles_axis_vector(self):
        self.alpha = compute_alpha(self.axis_vector)
        self.beta = compute_beta(self.axis_vector)
        self.gamma = compute_gamma(self.x_vector, self.y_vector)

    def compute_axis_centre(self) -> np.ndarray:
        """
        Calculate the center point of the axis.

        Returns:
            np.ndarray: The center point of the axis.
        """
        assert self.axis_points is not None,\
              "Axis points must be set before getting the center."
        return np.mean(self.axis_points, axis=0)


    # Asserters

    def assert_axis_points(self, string=None):
        if self.axis_points is None:
            raise AssertionError("Axis points must be set. " + string)
            
    def assert_z_vector(self, string=None):
        if self.z_vector is None:
            raise AssertionError("z vector must be set. " + string)

    def assert_x_vector(self, string=None):
        if self.x_vector is None:
            raise AssertionError("x vector must be set. " + string)
    
    def assert_y_vector(self, string=None):
        if self.y_vector is None:
            raise AssertionError("y vector must be set. " + string)
        
    def assert_xyz_vectors(self, string=None):
        self.assert_x_vector(string)
        self.assert_y_vector(string)
        self.assert_z_vector(string)
        
    def assert_alpha(self, string=None):
        if self.alpha is None:
            raise AssertionError("Alpha must be set. " + string)
        
    def assert_beta(self, string=None):
        if self.beta is None:
            raise AssertionError("Beta must be set. " + string)
        
    def assert_gamma(self, string=None):
        if self.gamma is None:
            raise AssertionError("Gamma must be set. " + string)
        
    # Open3d Visualization
    def draw_points(self, vis: o3d.visualization.Visualizer, color: np.ndarray = K):
        self.init_pcd()
        self.pcd.paint_uniform_color(color)
        vis.add_geometry(self.pcd)

    def draw_axis(self, vis: o3d.visualization.Visualizer, length=2000, color: np.ndarray = K):
        self.assert_axis_points("In draw_axis()")

        center = self.get_axis_centre()

        axis = generate_line(center, self.axis_vector, length)
        axis.paint_uniform_color(color)
        
        vis.add_geometry(axis)

    def draw_xyz_vectors(self, vis: o3d.visualization.Visualizer, length=1000, colors= [G, B, R]):
        self.assert_xyz_vectors("In draw_xyz_vectors()")

        center = self.get_axis_centre()

        x_axis = generate_line(center, self.x_vector, length)
        x_axis.paint_uniform_color(colors[0])
        
        y_axis = generate_line(center, self.y_vector, length)
        y_axis.paint_uniform_color(colors[1])

        z_axis = generate_line(center, self.z_vector, length)
        z_axis.paint_uniform_color(colors[2])

        vis.add_geometry(x_axis)
        vis.add_geometry(y_axis)
        vis.add_geometry(z_axis)

    def draw_default(self, vis: o3d.visualization.Visualizer):
        self.draw_points(vis)
        self.draw_axis(vis)

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

        

        




    
class MuscleMap():
    
    name: str # Name of the muscle map

    muscles: list # List of Muscle objects
    
    axis_points: np.ndarray # Axis points of the muscle map, shape: (n, 3)

    axis_vector: np.ndarray # Axis vector of the muscle map, shape: (3,), also roll vector axis

    x_vector: np.ndarray # x-axis of the relative coordinate system of the muscle map, same as axis_vector

    y_vector: np.ndarray # y-axis of the relative coordinate system of the muscle map

    z_vector: np.ndarray # z-axis of the relative coordinate system of the muscle map

    roll_point: np.ndarray # Roll reference point of the muscle map

    roll_vector: np.ndarray # Roll vector of the muscle map

    

    
    def __init__(self, muscles: list, axis_points: np.ndarray=None, name=None, roll=0, compute_dependend_attributes=True):
        self.muscles = muscles

        self.roll = roll

        if np.any(axis_points, None):
            self.set_axis_points(axis_points, compute_dependend_attributes=compute_dependend_attributes)

        else:
            self.axis_points = None
            self.axis_vector = None

        if name is None:
            name = "MuscleMap"

        self.name = name

        

    @classmethod
    def from_directory(cls, dir_path: Path, nb_muscles:int=5, axis_points: np.ndarray=None, name=None, roll=0):
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

    def compute_axis_vector(self) -> np.ndarray:
        return compute_vector_from_points(self.axis_points[0], self.axis_points[1])
    
    def compute_self_yaw(self) -> float:
        return compute_yaw(self.axis_vector)
    
    def compute_self_pitch(self) -> float:
        return compute_pitch(self.axis_vector)
    
    def compute_set_yaw_pitch(self):
        self.yaw = self.compute_self_yaw()
        self.pitch = self.compute_self_pitch()

    # Translation
    def translate(self, translation: np.ndarray, new_name=None):
        translated_muscles = [muscle.translate(translation) for muscle in self.muscles]
        translated_axis_points = self.axis_points + translation

        if new_name:
            name = new_name
        else:
            name = self.name

        return MuscleMap(translated_muscles, translated_axis_points, name=name)
    
    def centered_on_axis_point(self):
        return self.translate(-self.axis_points[0])
  
    # Rotation

    def rotate(self, rotvec: np.ndarray, theta: float, new_name=None):
        
        assert self.axis_points is not None, "Axis points must be set before rotating."

        rotated_muscles = [muscle.rotate(rotvec, theta) for muscle in self.muscles]

        new_axis_points = rotated_muscles[0].axis_points

        if new_name:
            name = new_name
        else:
            name = self.name + "_rotated"
        
        return MuscleMap(rotated_muscles, axis_points=new_axis_points, name=name)
        
    def to_yaw1(self, yaw: float):
        """
        Rotate the muscle map to a given yaw angle.

        Parameters:
            yaw (float): The yaw angle in radians.

        Returns:
            MuscleMap: A new MuscleMap object representing the rotated muscle map.
        """
        current_yaw = self.get_yaw()

        delta = yaw - current_yaw

        #vector of rotation z-axis
        rotvec = np.array([0, 0, 1])

        m = self.rotate(rotvec, delta)
        m.roll = self.roll
        return m
    
    def to_yaw(self, yaw: float, print_info=False):
        return self.to_yaw2(yaw, print_info=print_info)
        
    
    def to_yaw2(self, yaw: float, print_info=False):
        """
        Rotate the muscle map to a given yaw angle.

        Parameters:
            yaw (float): The yaw angle in radians.

        Returns:
            MuscleMap: A new MuscleMap object representing the rotated muscle map.
        """
        current_pitch= self.get_pitch()
        current_yaw = self.get_yaw()

        delta = yaw - current_yaw

        # perpendicular vector to axis in x-z plane
        rotvec = np.array([0,0,1])

        m = self
        
        m = m.rotate(rotvec, delta)

        yaw_after1 = m.get_yaw()
        pitch_after1 = m.get_pitch()


        m = m.to_pitch(current_pitch)

        yaw_after2 = m.get_yaw()
        pitch_after2 = m.get_pitch()
        if print_info:

            print('current pitch: ', current_pitch)
            print('current yaw: ', current_yaw)

            print('new yaw before to_pitch(): ', yaw_after1)
            print('new pitch before to_pitch(): ', pitch_after1)

            print('new yaw after to_pitch(): ', yaw_after2)
            print('new pitch after to_pitch(): ', pitch_after2)

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
        current_pitch = self.get_pitch()

        delta = pitch - current_pitch

        #create perpendicular vector to axis vector in x-y plane
        rotvec = np.array([self.axis_vector[1], -self.axis_vector[0], 0])
        rotvec = rotvec / linalg.norm(rotvec)

        m = self.rotate(rotvec, delta)
        m.roll = self.roll
        return m
    
    def to_yaw_pitch(self, yaw: float, pitch: float):
        """
        Rotate the muscle map to a given yaw and pitch angle.

        Parameters:
            yaw (float): The yaw angle in radians.
            pitch (float): The pitch angle in radians.

        Returns:
            MuscleMap: A new MuscleMap object representing the rotated muscle map.
        """
        m = self.to_yaw(yaw)
        m = m.to_pitch(pitch)
        return m
    

    # Scaling  
    def scale(self, scale_factor: float):
            """
            Scales the muscles in the MuscleMap by the given scale factor.

            Args:
                scale_factor (float): The factor by which to scale the muscles.

            Returns:
                MuscleMap: A new MuscleMap object with scaled muscles.
            """
            scaled_muscles = [muscle.scale(scale_factor) for muscle in self.muscles]
            scaled_axis_points = scaled_muscles[0].get_axis_points()
            
            return MuscleMap(scaled_muscles, axis_points=scaled_axis_points, name=self.name)

    # Open3D visualization
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

    # Getters

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


    def get_muscles(self):
        return self.muscles
            
    def get_map2d(self):
        individual_maps = [muscle.get_map2d() for muscle in self.muscles]
        return Map2d(individual_maps, axis_points=self.axis_points[:, :2])
    
    def get_points(self):
        return np.vstack([muscle.points for muscle in self.muscles])
    
    def get_yaw(self):
        return self.yaw
    
    def get_pitch(self):
        return self.pitch

    def get_roll(self):
        return self.roll
    
    def get_name(self):
        return self.name
    
    def get_x_axis_vector(self):
        return self.x_vector
    
    def get_y_axis_vector(self):
        return self.y_vector
    
    def get_z_axis_vector(self):
        return self.z_vector
    
    def get_relative_frame_vectors(self):
        return self.x_vector, self.y_vector, self.z_vector

    # Setters

    def set_axis_points(self, axis_points: np.ndarray, compute_dependend_attributes=True):
        self.axis_points = axis_points

        if compute_dependend_attributes:
            self.axis_vector = self.compute_axis_vector()
            self.yaw = self.compute_self_yaw()
            self.pitch = self.compute_self_pitch()

            self.init_angles_vectors()

            for muscle in self.muscles:
                muscle.set_axis_points(axis_points, compute_dependend_attributes=compute_dependend_attributes)

    def init_x_vector(self):
        self.x_vector = self.axis_vector/np.linalg.norm(self.axis_vector)

    def init_y_vector(self):
        assert self.roll ==0 , "Roll must be set to zero before initializing y vector."

        # y vector is perpendicular to x vector. 
        # If roll = 0, y vector lies in x-y plane
        z_axis = np.array([0, 0, 1])
        y_vector = np.cross(self.x_vector, z_axis)

        self.y_vector = y_vector/np.linalg.norm(y_vector)

    def init_z_vector(self):

        # z vector is perpendicular to x and y vector
        z_vector = np.cross(self.x_vector, self.y_vector)
        self.z_vector = z_vector/np.linalg.norm(z_vector)

    def init_relative_frame_vectors(self):
        self.init_x_vector()
        self.init_y_vector()
        self.init_z_vector()

    def set_name(self, name: str):
        self.name = name
    
    def set_default_axis_points(
            self, 
            direction: np.ndarray=np.array([0.4,0.6,-0.3]), 
            dist: float=700
            ):
        com = self.get_com()
        axis_points = np.array([
            com - dist*direction,
            com + dist*direction
        ])
        self.set_axis_points(axis_points)

    def reset_roll(self):
        self.assert_angles_axis_vector("reset_roll()")

        # roll is 0 when pitch axis is in the x-y plane
        # and when

        pass





    # Asserter
    def assert_angles_axis_vector(self, string=None):
        assert self.roll_axis_vector is not None, \
            "Roll axis vector not set. " + string
        assert self.pitch_axis_vector is not None, \
            "Pitch axis vector must be set. " + string
        assert self.yaw_axis_vector is not None, \
            "Yaw axis vector must be set. " + string
        

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
    

    def translate(self, translation_vec: np.ndarray):
        translated_points = self.points + translation_vec
        translated_axis_points = self.axis_points + translation_vec
        return IndividualMap2d(translated_points, axis_points=translated_axis_points, name=self.name)
    
    def get_points(self, d3=False):
        if d3:
            return np.hstack((self.points, np.zeros((self.points.shape[0], 1))))
        else:
            return self.points
    
    def get_name(self):
        return self.name
    
    def scale(self, scale_factor: float, warning=True): 
        if not np.allclose(self.axis_points[0], np.zeros(2)) and warning:
            print("Map: axis_points[0] not at origin, scaling might not be accurate.")

        scaled_points = self.points * scale_factor
        scaled_axis_points = self.axis_points * scale_factor
        return IndividualMap2d(scaled_points, axis_points=scaled_axis_points, name=self.name)
    

            
    def set_axis_points(self, axis_points: np.ndarray):
        self.axis_points = axis_points

    def set_name(self, name: str):
        self.name = name    


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
    
    def translate(self, translation_vec: np.ndarray):
        translated_maps = [mmap.translate(translation_vec) for mmap in self.individual_maps]
        translated_axis_points = self.axis_points + translation_vec
        return Map2d(translated_maps, axis_points=translated_axis_points)
    
    def scale(self, scale_factor: float, warning=True):
        scaled_maps = [mmap.scale(scale_factor, warning=warning) for mmap in self.individual_maps]
        scaled_axis_points = self.axis_points * scale_factor
        return Map2d(scaled_maps, axis_points=scaled_axis_points)

    # Plotting

    def plot_axis(self, ax, **kwargs):
        ax.plot(self.axis_points[:, 0], self.axis_points[:, 1], **kwargs)
        return ax
    
    def plot_maps(self, ax, colors=None, **kwargs):
        if colors is None:
            colors = get_equally_spaced_colors(len(self.individual_maps))

        for mmap, color in zip(self.individual_maps, colors):
            ax.scatter(mmap.get_points()[:, 0], mmap.get_points()[:, 1], color=color, **kwargs)
        return ax

    # Getters

    def get_maps(self):
        return self.individual_maps
    
    def get_axis_points(self):
        return self.axis_points
    
    def get_points(self, d3=False):
        points = [m.get_points(d3=d3) for m in self.individual_maps]
        return np.vstack(points)

    
    # Setters

    def set_axis_points(self, axis_points: np.ndarray):
        self.axis_points = axis_points
        for m in self.individual_maps:
            m.set_axis_points(axis_points)    


    
    