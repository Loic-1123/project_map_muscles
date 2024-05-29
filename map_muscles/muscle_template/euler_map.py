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


# Using Euler angles: https://en.wikipedia.org/wiki/Euler_angles#Classic_Euler_angles
pi = np.pi
R, G, B = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
K, W = np.array([0,0,0]), np.array([1,1,1])

def assert_unit_vector(vector: np.ndarray, string=None):
    if not np.isclose(linalg.norm(vector), 1):
        raise AssertionError(f"Vector must be a unit vector. Got {vector}. {string}")
    
def assert_not_aligned_with_z(vector: np.ndarray, string=None):
    if np.allclose([vector[0], vector[1]],  0):
        raise AssertionError(f"Vector must not be aligned with the z-axis. {string}")

def assert_angle(truth, angle, tol=1e-7, assert_str=None):
    assert np.isclose(truth, angle, atol=tol)\
        or np.isclose(truth, angle%(2*pi), atol=tol)\
        or np.isclose(truth%(2*pi), angle%(2*pi), atol=tol)\
        or np.isclose(truth, angle+2*pi, atol=tol)\
        or np.isclose(truth, angle-2*pi, atol=tol), assert_str

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

        warning_str = "Warning: In euler_map.py::compute_alpha(): If z3 is 1 or -1, beta=0 or pi, Alpha is confounded with Gamma. Returns alpha=0."
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

    z3 = z_vector[2]

    if np.isclose(z3, 1, atol=1e-07):
        beta = 0

    elif np.isclose(z3, -1, atol=1e-07):
        beta = np.pi
    else:
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
        warning_str = "Warning: In euler_map.py::compute_gamma(): If y3 is 0, Beta = 0, hence, alpha and gamma angles are confounded. Returns angle between absolute x-axis given x_vector."
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
            self.set_axis_points(axis_points, gamma=gamma, compute_dependent_attributes=True)

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

    # Transformations
    ## Translation
    def translate(self, translation: np.ndarray):
        """
        Translates the muscle by adding the given translation vector to its points and axis points.

        Parameters:
            translation (np.ndarray): The translation vector to be added to the muscle's points and axis points.

        Returns:
            Muscle: A new Muscle object with translated points and axis points.
        """
        t_points = self.points + translation
        t_axis_points = self.axis_points + translation
        return Muscle(t_points, name=self.name, axis_points=t_axis_points)
    
    def centered_on_axis_point(self):
        """
        Translates the muscle template to be centered on the first axis point.

        Returns:
            The translated muscle template.
        """
        return self.translate(-self.axis_points[0])
    
    ## Rotation: rotation occurs around the first axis point
    def reset_rotation(self, print_warning=False, raise_assertions=False, translate_back=True):
        """
        Resets the rotation of the muscle (alpha=beta=gamma = 0).

        Args:
            raise_assertions (bool, optional): Whether to raise assertions to check the correctness of the reset rotation. 
                                               Defaults to False.

        Returns:
            Muscle: A new Muscle object with the rotation reset.

        """
        # translate to origin
        m = self.centered_on_axis_point()
        
        m.assert_angles("In reset_rotation()")
        alpha, beta, gamma = m.get_angles()
        x, y, z  = m.get_vectors()

        r = Rot.from_euler('ZXZ', [-gamma, -beta, -alpha])
        r_points = r.apply(m.points)
        r_axis_points = r.apply(m.axis_points)


        if raise_assertions:
            rx, ry, rz = r.apply(x), r.apply(y), r.apply(z)

            ralpha = compute_alpha(rz, print_warning)
            rbeta = compute_beta(rz)
            rgamma = compute_gamma(rx, ry, print_warning)

            assert np.isclose(ralpha, 0), f"Alpha not 0 after reset_rotation. Got {ralpha}."
            assert np.isclose(rbeta, 0), f"Beta not 0 after reset_rotation. Got {rbeta}."
            assert np.isclose(rgamma, 0), f"Gamma not 0 after reset_rotation. Got {rgamma}."

            assert np.allclose(rx, np.array([1,0,0])), f"X vector not [1,0,0] after reset_rotation. Got {rx}."
            assert np.allclose(ry, np.array([0,1,0])), f"Y vector not [0,1,0] after reset_rotation. Got {ry}."
            assert np.allclose(rz, np.array([0,0,1])), f"Z vector not [0,0,1] after reset_rotation. Got {rz}."


        # translate back
        if translate_back:
            r_points = r_points + self.axis_points[0]
            r_axis_points = r_axis_points + self.axis_points[0]

        return Muscle(r_points, name=self.name, axis_points=r_axis_points, gamma=0)
    
    def rotate_to_angles(self, angles: np.ndarray, raise_assertions=False, translate_back=True, print_warning=False, print_debug_info=False):
        """
        Rotate the muscle to the specified angles.

        Args:
            angles (np.ndarray): Array of three angles [alpha, beta, gamma] in radians.
            raise_assertions (bool, optional): Whether to raise assertions to check the correctness of the rotation. Defaults to False.
            translate_back (bool, optional): Whether to translate the muscle back to its original position after rotation. Defaults to True.

        Returns:
            Muscle: The rotated muscle.

        Raises:
            AssertionError: If raise_assertions is True and the resulting angles after rotation do not match the specified angles.

        """
        rm = self.reset_rotation(translate_back=False, raise_assertions=True)

        r = Rot.from_euler('ZXZ', angles)

        rpoints = r.apply(rm.get_points())
        raxis = r.apply(rm.get_axis_points())

        if raise_assertions:
            x, y, z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
            rx, ry, rz = r.apply(x), r.apply(y), r.apply(z)

            axis_vector_expecte_str = f'Expected axis after rotation vector to be rx: {rx}, ry: {ry}, rz: {rz}'\
            + '\n' + f'Check if standard unit vector basis: reset_vectors = {rm.get_vectors()}'

            alpha, beta, gamma = angles
            
            ralpha = compute_alpha(rz, print_warning)
            rbeta = compute_beta(rz)
            rgamma = compute_gamma(rx, ry, print_warning)

            angles_expected_str = f'Expected angles to be alpha: {alpha}, beta: {beta}, gamma: {gamma} but got {ralpha, rbeta, rgamma}, '\
            + '\n'+ f'computed from rotated unit vectors: rx: {rx}, ry: {ry}, rz: {rz}'

            if print_debug_info:
                print(f'alpha: {alpha}', f'beta: {beta}', f'gamma: {gamma}', \
                       f'After rotation: alpha: {ralpha}, beta: {rbeta}, gamma: {rgamma}')
            
            added_str = '\n' + axis_vector_expecte_str + '\n' + angles_expected_str

            assert_alpha_str = f"Alpha not {alpha} after rotate_to_angles(). Got {ralpha}." + added_str
            assert_gamma_str = f"Gamma not {gamma} after rotate_to_angles(). Got {rgamma}." + added_str
            
            if np.isclose(beta, 0) or np.isclose(beta, pi):
                assert_angle(ralpha, 0, assert_str=assert_alpha_str)
                assert_angle(rgamma, gamma+alpha, assert_str=assert_gamma_str)
            else: 
                assert_angle(ralpha, alpha, assert_str=assert_alpha_str)
                assert_angle(rgamma, gamma, assert_str=assert_gamma_str)

            assert np.isclose(rbeta, beta), f"Beta not {beta} after rotate_to_angles(). Got beta={rbeta}."\
            + added_str

        if translate_back:
            ref_axis_point = self.get_axis_points()[0]
            rpoints = rpoints + ref_axis_point
            raxis = raxis + ref_axis_point

        # if confounded alpha and gamma (beta = 0 or pi), set gamma to alpha + gamma, alpha = 0

        new_gamma = angles[2]
        if np.isclose(angles[1], 0):
            new_gamma = angles[2] + angles[0]
        if np.isclose(angles[1], pi):
            new_gamma = angles[2] - angles[0]
        
        return Muscle(rpoints, name=self.name, axis_points=raxis, gamma=new_gamma)

    def apply_rotation(self, r: Rot, translate_back=True):
        """
        Apply a rotation to the muscle.

        Args:
            r (Rot): The rotation to apply.
            translate_back (bool, optional): Whether to translate the muscle back to its original position after rotation. Defaults to True.

        Returns:
            Muscle: The rotated muscle.
        """
        centered = self.centered_on_axis_point()

        r_points = r.apply(centered.get_points())
        r_axis_points = r.apply(centered.get_axis_points())

        if translate_back:
            ref_axis_point = self.get_axis_points()[0]
            r_points = r_points + ref_axis_point
            r_axis_points = r_axis_points + ref_axis_point

        new_gamma = compute_gamma(r.apply(centered.get_x_vector()), r.apply(centered.get_y_vector()), print_warning=False)

        return Muscle(r_points, name=self.name, axis_points=r_axis_points, gamma=new_gamma)

    ## Scaling        
    def scale(self, scale_factor: float):
            """
            Scales the muscle by the given scale factor.
            Relative to the axis point: Position of the first axis point is preserved.

            Args:
                scale_factor (float): The factor by which to scale the muscle.

            Returns:
                Muscle: The scaled muscle.
            """

            ref = self.axis_points[0]

            centered_Muscle = self.centered_on_axis_point()

            scaled_points = centered_Muscle.get_points() * scale_factor
            scaled_axis_points = centered_Muscle.get_axis_points() * scale_factor

            # translate back
            scaled_points = scaled_points + ref
            scaled_axis_points = scaled_axis_points + ref

            return Muscle(scaled_points, name=self.name, axis_points=scaled_axis_points, gamma=self.gamma)
    
    ## Projection
    def project_points_on_xy_plane(self, remove_z_axis=False):

        points = self.points[:, :2]
        
        if remove_z_axis:
            return points
        else:
            return np.hstack((points, np.zeros((points.shape[0], 1))))
    
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
    
    def get_vectors(self):
        return self.x_vector, self.y_vector, self.z_vector
    
    def get_alpha(self):
        return self.alpha
    
    def get_beta(self):
        return self.beta
    
    def get_gamma(self):
        return self.gamma
    
    def get_angles(self):
        return self.alpha, self.beta, self.gamma
    
    def get_pcd(self):
        assert self.pcd is not None, "Point cloud data not initialized."
        return self.pcd

    
    # Setters / Initializers
      
    def set_axis_points(self, axis_points: np.ndarray, gamma=0, compute_dependent_attributes=True):
        """
        Set the axis points of the muscle template.

        Parameters:
        - axis_points (np.ndarray): The array of axis points.
        - gamma (float): The gamma value.
        - compute_dependent_attributes (bool): Whether to compute dependent attributes.
        - - dependent attributes: z_vector, alpha, beta. x_vector, y_vector, gamma are set to default values (gamma=0).

        Returns:
        None
        """

        self.axis_points = axis_points

        if compute_dependent_attributes:
            self.init_z_vector()
            self.init_aplha_from_z_vector()
            self.init_beta_from_z_vector()
            self.init_gamma_and_x_y_vectors(gamma=gamma)

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

    def init_aplha_from_z_vector(self, print_warning=False, raise_warning=False):
        self.assert_z_vector("In init_alpha()")
        self.alpha = compute_alpha(self.z_vector, print_warning, raise_warning)

    def init_beta_from_z_vector(self):
        self.assert_z_vector("In init_beta()")
        self.beta = compute_beta(self.z_vector)

    def init_gamma_and_x_y_vectors(self, gamma=0):
        self.assert_alpha("In init_gamma_and_x_y_vectors()")
        self.assert_beta("In init_gamma_and_x_y_vectors()")
        self.assert_z_vector("In init_gamma_and_x_y_vectors()")

        self.gamma = gamma

        r = Rot.from_euler('ZXZ', [self.alpha, self.beta, self.gamma])
        self.x_vector = r.apply(np.array([1,0,0]))
        self.y_vector = r.apply(np.array([0,1,0]))

    def init_default_axis_points(
        self, 
        direction: np.ndarray=np.array([0.4,0.6,-0.3]), 
        dist: float=600,
        gamma=0
        ):
        com = self.compute_com()
        axis_points = np.array([
            com + dist*direction,
            com - dist*direction
        ])
        self.set_axis_points(axis_points, gamma=gamma)

    def init_pcd(self):
        if self.pcd is not None:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd = pcd

    # Computers

    def compute_z_vector(self) -> np.ndarray:
        z_vec = compute_normal_vector_from_points(self.axis_points[0], self.axis_points[1])
        return z_vec
    
    def compute_axis_centre(self) -> np.ndarray:
        """
        Calculate the center point of the axis.

        Returns:
            np.ndarray: The center point of the axis.
        """
        assert self.axis_points is not None,\
              "Axis points must be set before getting the center."
        return np.mean(self.axis_points, axis=0)
    
    def compute_com(self) -> np.ndarray:
        """
        Calculate the center of mass of the muscle.

        Returns:
            np.ndarray: The center of mass of the muscle.
        """
        return np.mean(self.points, axis=0)

    def compute_projection_on_xy_plane(self, remove_z_axis=False):
        points = self.points[:, :2]
        
        if remove_z_axis:
            return points
        else:
            return np.hstack((points, np.zeros((points.shape[0], 1))))
        
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
        
    def assert_angles(self, string=None):
        self.assert_alpha(string)
        self.assert_beta(string)
        self.assert_gamma(string)
        
    # Open3d Visualization
    def draw_points(self, vis: o3d.visualization.Visualizer, color: np.ndarray = K):
        self.init_pcd()
        self.pcd.paint_uniform_color(color)
        vis.add_geometry(self.pcd)

    def draw_axis(self, vis: o3d.visualization.Visualizer, length=2000, color: np.ndarray = K):
        self.assert_axis_points("In draw_axis()")

        center = self.compute_axis_centre()

        axis = generate_line(center, self.z_vector, length)
        axis.paint_uniform_color(color)
        
        vis.add_geometry(axis)

    def draw_xyz_vectors(self, vis: o3d.visualization.Visualizer, length=500, colors= [R, G, B]):
        self.assert_xyz_vectors("In draw_xyz_vectors()")

        center = self.get_axis_points()[0]

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
        self.draw_xyz_vectors(vis)

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

    alpha: float # Alpha angle of the muscle, in radians

    beta: float # Beta angle of the muscle, in radians

    gamma: float # Gamma angle of the muscle, in radians

    x_vector: np.ndarray # x-axis of the relative coordinate system of the muscle

    y_vector: np.ndarray # y-axis of the relative coordinate system of the muscle

    z_vector: np.ndarray # z-axis of the relative coordinate system of the muscle. 
    #Also axis vector. The muscle axis is assumed to be the z-axis of the relative coordinate system.

    
    def __init__(
            self, 
            muscles: list, 
            axis_points: np.ndarray=None, 
            name=None, gamma=0, 
            compute_dependent_attributes=True
        ):
        
        self.muscles = muscles

        self.gamma = gamma

        if np.any(axis_points, None):
            self.set_axis_points(axis_points, compute_dependent_attributes=compute_dependent_attributes)

        else:
            self.axis_points = None
            self.set_xyz_vectors_to_None()
            self.set_angles_to_None()

        if name is None:
            name = "MuscleMap"

        self.name = name

    @classmethod
    def from_directory(
        cls, dir_path: Path, nb_muscles:int=5, 
        axis_points: np.ndarray=None, 
        name=None, gamma=0, 
        compute_dependent_attributes=True
        ):
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

        muscles = [Muscle.from_array_file(file) for file in files]

        if name is None:
            name = dir_path.stem

        return cls(
            muscles, 
            axis_points=axis_points, 
            name=name, gamma=gamma, 
            compute_dependent_attributes=compute_dependent_attributes
            )


    # Transformations
    ## Translation
    def translate(self, translation_vec: np.ndarray):
        """
        Translates the muscle map by a given translation vector.

        Args:
            translation_vec (np.ndarray): The translation vector to apply.

        Returns:
            MuscleMap: The translated muscle map.

        """
        t_muscles = [muscle.translate(translation_vec) for muscle in self.muscles]
        t_axis = self.axis_points + translation_vec
        return MuscleMap(t_muscles, axis_points=t_axis, name=self.name, gamma=self.gamma)

    def centered_on_axis_point(self):
        """
        Translates the muscle map to be centered on the first axis point.
        That is, the first axis point will be at the origin.

        Returns:
            Translated muscle template.
        """
        return self.translate(-self.axis_points[0])
    
    ## Rotation around first axis point
    def reset_rotation(self, print_warning=False, raise_assertions=False, translate_back=True):
        self.assert_angles("In reset_rotation()")
        
        r_muscles = [muscle.reset_rotation(
                print_warning=print_warning, 
                raise_assertions=raise_assertions, translate_back=translate_back) \
                for muscle in self.muscles]
        
        r = Rot.from_euler('ZXZ', [-self.gamma, -self.beta, -self.alpha])
        centered_axis = self.get_axis_points() - self.get_axis_points()[0]
        r_axis_points = r.apply(centered_axis)

        if translate_back:
            r_axis_points = r_axis_points + self.axis_points[0] 

        return MuscleMap(r_muscles, axis_points=r_axis_points, name=self.name, gamma=0)
    
    def rotate_to_angles(
            self, 
            angles: np.ndarray, 
            raise_assertions=False, translate_back=True, 
            print_warning=False, print_debug_info=False
        ):

        rmmap = self.reset_rotation(translate_back=translate_back, raise_assertions=True)

        r_muscles = [muscle.rotate_to_angles(
            angles, raise_assertions=raise_assertions, 
            translate_back=translate_back, print_warning=print_warning, print_debug_info=print_debug_info) \
            for muscle in rmmap.muscles]

        r = Rot.from_euler('ZXZ', angles)     
        centered_axis = rmmap.get_axis_points() - rmmap.get_axis_points()[0]   
        r_axis_points = r.apply(centered_axis)

        if translate_back:
            r_axis_points = r_axis_points + self.axis_points[0]

        # if confounded alpha and gamma (beta = 0 or pi), set gamma to alpha + gamma, alpha = 0
        new_gamma = angles[2]
        if np.isclose(angles[1], 0):
            new_gamma = angles[2] + angles[0]
        if np.isclose(angles[1], pi):
            new_gamma = angles[2] - angles[0]

        return MuscleMap(r_muscles, axis_points=r_axis_points, name=self.name, gamma=new_gamma)

    def apply_rotation(self, r: Rot, translate_back=True):
        """
        Apply a rotation to the muscle map.

        Args:
            r (Rot): The rotation to apply.
            translate_back (bool, optional): Whether to translate the muscle back to its original position after rotation. Defaults to True.

        Returns:
            MuscleMap: The rotated muscle map.
        """
        r_muscles = [muscle.apply_rotation(r, translate_back=translate_back) for muscle in self.muscles]


        ref_point = self.axis_points[0]
        caxis = self.axis_points - ref_point
        r_axis_points = r.apply(caxis)

        if translate_back:
            r_axis_points = r_axis_points + ref_point

        new_gamma = compute_gamma(r.apply(self.get_x_vector()), r.apply(self.get_y_vector()), print_warning=False)

        return MuscleMap(r_muscles, axis_points=r_axis_points, name=self.name, gamma=new_gamma)


    ## Scaling

    def scale(self, scale_factor: float):
        """
        Scales the muscle map by the given scale factor.

        Args:
            scale_factor (float): The factor by which to scale the muscle map.

        Returns:
            MuscleMap: The scaled muscle map.

        """
        ref = self.axis_points[0]

        centered_MuscleMap = self.centered_on_axis_point()

        scaled_muscles = [muscle.scale(scale_factor) for muscle in centered_MuscleMap.muscles]
        scaled_axis_points = centered_MuscleMap.axis_points * scale_factor

        # translate back
        scaled_axis_points = scaled_axis_points + ref

        return MuscleMap(scaled_muscles, axis_points=scaled_axis_points, name=self.name, gamma=self.gamma)    

    # Getters
    def get_muscles(self):
        return self.muscles
    
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
    
    def get_vectors(self):
        return self.x_vector, self.y_vector, self.z_vector
    
    def get_alpha(self):
        return self.alpha
    
    def get_beta(self):
        return self.beta
    
    def get_gamma(self):
        return self.gamma
    
    def get_angles(self):
        return self.alpha, self.beta, self.gamma

    # Setters

    def set_axis_points(self, axis_points: np.ndarray, gamma=0, compute_dependent_attributes=True):
        """
        Set the axis points of the muscle map.

        Parameters:
        - axis_points (np.ndarray): The array of axis points.
        - gamma (float): The gamma value.
        - compute_dependent_attributes (bool): Whether to compute dependent attributes.
        - - dependent attributes: z_vector, alpha, beta. x_vector, y_vector, gamma are set to default values (gamma=0).

        Returns:
        None
        """
        
        self.axis_points = axis_points

        if compute_dependent_attributes:
            self.init_z_vector()
            self.init_aplha_from_z_vector()
            self.init_beta_from_z_vector()
            self.init_gamma_and_x_y_vectors(gamma=gamma)
            
            self.assert_muscles("In set_axis_points()")
            self.init_muscles_axis_points()

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

    def init_aplha_from_z_vector(self, print_warning=False, raise_warning=False):
        self.assert_z_vector("In init_alpha()")
        self.alpha = compute_alpha(self.z_vector, print_warning, raise_warning)

    def init_beta_from_z_vector(self):
        self.assert_z_vector("In init_beta()")
        self.beta = compute_beta(self.z_vector)

    def init_gamma_and_x_y_vectors(self, gamma=0):
        self.assert_alpha("In init_default_gamma_and_x_y_vectors()")
        self.assert_beta("In init_default_gamma_and_x_y_vectors()")
        self.assert_z_vector("In init_default_gamma_and_x_y_vectors()")

        self.gamma = gamma

        r = Rot.from_euler('ZXZ', [self.alpha, self.beta, self.gamma])
        self.x_vector = r.apply(np.array([1,0,0]))
        self.y_vector = r.apply(np.array([0,1,0]))

    def init_muscles_axis_points(self):
        self.assert_axis_points("In init_muscles_axis_points()")
        for muscle in self.muscles:
            muscle.set_axis_points(self.axis_points, gamma=self.gamma)

    def init_default_axis_points(
        self, 
        direction: np.ndarray=np.array([0.4,0.6,-0.3]), 
        dist: float=900,
        gamma=0
        ):
        com = self.compute_com()
        axis_points = np.array([
            com + dist*direction,
            com - dist*direction
        ])
        self.set_axis_points(axis_points, gamma=gamma)

    # Computers

    def compute_z_vector(self) -> np.ndarray:
        z_vec = compute_normal_vector_from_points(self.axis_points[0], self.axis_points[1])
        return z_vec
    
    def compute_axis_centre(self) -> np.ndarray:
        """
        Calculate the center point of the axis.

        Returns:
            np.ndarray: The center point of the axis.
        """
        assert self.axis_points is not None,\
              "Axis points must be set before getting the center."
        return np.mean(self.axis_points, axis=0)
    
    def compute_com(self) -> np.ndarray:
        """
        Calculate the center of mass of the muscle.

        Returns:
            np.ndarray: The center of mass of the muscle.
        """
        points = []

        for muscle in self.muscles:
            points.extend(muscle.get_points())

        return np.mean(points, axis=0)
    
    def compute_axis_vector(self, unit=False) -> np.ndarray:
        """
        Calculate the axis vector of the muscle map.

        Args:
            unit (bool, optional): Whether to return the unit vector. Defaults to False.

        Returns:
            np.ndarray: The axis vector of the muscle map.
        """
        axis_vector = self.axis_points[1] - self.axis_points[0]
        if unit:
            axis_vector = axis_vector / np.linalg.norm(axis_vector)
        return axis_vector
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
        
    def assert_angles(self, string=None):
        self.assert_alpha(string)
        self.assert_beta(string)
        self.assert_gamma(string)

    def assert_muscles(self, string=None):
        if self.muscles is None:
            raise AssertionError("Muscles must be set. " + string)


    # Open3d Visualization
    def draw_points(self, vis: o3d.visualization.Visualizer, colors: list = None):
        if colors is None:
            colors = get_equally_spaced_colors(len(self.muscles))
        
        for muscle, c in zip(self.muscles, colors):
            muscle.draw_points(vis, color=c)

    def draw_axis(self, vis: o3d.visualization.Visualizer, length=2000, color: np.ndarray = K):
        self.assert_axis_points("In draw_axis()")

        center = self.compute_axis_centre()

        axis = generate_line(center, self.z_vector, length)
        axis.paint_uniform_color(color)
        
        vis.add_geometry(axis)

    def draw_xyz_vectors(self, vis: o3d.visualization.Visualizer, length=500, colors= [R, G, B]):
        self.assert_xyz_vectors("In draw_xyz_vectors()")

        center = self.get_axis_points()[0]

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
        self.draw_xyz_vectors(vis)