from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pathlib import Path
import cv2

import map_muscles.muscle_template.euler_map as mp
import map_muscles.extract_fluorescence.imaging_utils as imu

class MappedFrame():
    kin_img: np.ndarray
    kin_middle_axis: np.ndarray # 2D array of shape (2, 2) (middle view)
    kin_top_axis: np.ndarray # 2D array of shape (2, 2) (top view)

    mmap: mp.MuscleMap

    muscle_img: np.ndarray
    muscle_middle_axis: np.ndarray # 2D array of shape (2, 2) (middle view)

    # The different axes represent the trochanter and femur-tibia joints

    def __init__(
            self, 
            kin_img: np.ndarray, 
            kin_middle_axis:np.ndarray, 
            kin_top_axis:np.ndarray,
            mmap: mp.MuscleMap,
            muscle_img: np.ndarray = None,
            muscle_middle_axis: np.ndarray = None,
        ):
        self.set_img(kin_img)
        self.set_kin_middle_axis(kin_middle_axis)
        self.set_kin_top_axis(kin_top_axis)
        self.set_mmap(mmap)
        self.set_muscle_img(muscle_img)
        self.set_muscle_middle_axis(muscle_middle_axis)

    @classmethod
    def from_muscle_img_id(
        muscle_id: int, 
        kin_middle_axis: np.ndarray, 
        kin_top_axis: np.ndarray,
        mmap: mp.MuscleMap,
        path_to_muscle_img: Path,
        path_to_kin_img: Path,
        muscle_img_extension: str = 'tif',
        kin_to_muscle_div_factor: int = 4,
    ):
        """
        Create a MappedFrame object from a muscle image ID.
        Fetches the corresponding muscle image and its matching kin image.

        Args:
            muscle_id (int): The ID of the muscle image.
            kin_middle_axis (np.ndarray): The middle axis of the kin image.
            kin_top_axis (np.ndarray): The top axis of the kin image.
            mmap (mp.MuscleMap): The muscle map object.
            path_to_muscle_img (Path): The path to the muscle image directory.
            path_to_kin_img (Path): The path to the kin image directory.
            muscle_img_extension (str, optional): The extension of the muscle image files. Defaults to 'tif'.
            kin_to_muscle_div_factor (int, optional): The division factor between kin and muscle images. Defaults to 4.

        Returns:
            MappedFrame: The mapped frame object.
        """
        muscle_img = cv2.imread(str(path_to_muscle_img / f'{muscle_id:06d}.tif'), -1)

        muscle_min_id = imu.get_min_id(path_to_muscle_img, muscle_img_extension)
        kin_min_id = imu.get_min_id(path_to_kin_img, 'jpg') 

        kin_img = imu.get_matching_kin_img(
            muscle_frame_id=muscle_id, 
            min_id_muscle_file=muscle_min_id,
            ratio=kin_to_muscle_div_factor,
            min_id_kin_file=kin_min_id,
            kin_path=path_to_kin_img)
        
        return MappedFrame(
            kin_img=kin_img,
            kin_middle_axis=kin_middle_axis,
            kin_top_axis=kin_top_axis,
            mmap=mmap,
            muscle_img=muscle_img
            )
            
        
    # Transformations
    def align_map_axis_ref_point_on_kin(self):
        """
        Aligns the map axis reference points.

        This method aligns the x and y coordinates of the first map axis point
        with the first kinematic axis point.

        Returns:
            None
        """
        cmmap = self.mmap.centered_on_axis_point()
        tvec = np.array([self.kin_middle_axis[0][0], self.kin_middle_axis[0][1], 0])
        aligned_mmap = cmmap.translate(tvec)
        self.mmap = aligned_mmap

    def align_map_axis_ref_point_on_muscle(self):
        """
        Aligns the map axis reference points.

        This method aligns the x and y coordinates of the first map axis point
        with the first muscle axis point.

        Returns:
            None
        """
        cmmap = self.mmap.centered_on_axis_point()
        tvec = np.array([self.muscle_middle_axis[0][0], self.muscle_middle_axis[0][1], 0])
        aligned_mmap = cmmap.translate(tvec)
        self.mmap = aligned_mmap

    def scale_map(self, ratio):
        """
        Scales the map by a given ratio.
        Default: if ratio is None, compute the ratio between the kinematic axis and the map axis.

        Parameters:
            ratio (float): The ratio by which to scale the map.

        Returns:
            None
        """

        ref_point = self.mmap.get_axis_points()[0]

        cmmap = self.mmap.centered_on_axis_point()

        smmap = cmmap.scale(ratio)

        self.mmap = smmap.translate(ref_point)
    
    def orient_map_on_kin(self):
        """
        Orient the map to match the kinematic vector derived from the top and middle views axes.

        Returns:
            None
        """

        kin_vector = self.compute_kinematic_vector()

        # we want to align the axis vector with the kin_vector
        # for that, we extract the alpha and beta angles from the kin vector
        # and set the map to the corresponding angles
        alpha = mp.compute_alpha(kin_vector)
        beta = mp.compute_beta(kin_vector)

        # gamma stay unchanged
        gamma = self.mmap.get_gamma()

        self.mmap = self.mmap.rotate_to_angles([alpha, beta, gamma])
    
    def orient_map_on_muscle(self):
        """
        Orient the map to match the muscle vector derived from the top and middle views axes.

        Returns:
            None
        """

        muscle_vector = self.compute_muscle_vector(unit=True)

        # we want to align the axis vector with the kin_vector
        # for that, we extract the alpha and beta angles from the kin vector
        # and set the map to the corresponding angles
        alpha = mp.compute_alpha(muscle_vector)
        beta = mp.compute_beta(muscle_vector)

        # gamma stay unchanged
        gamma = self.mmap.get_gamma()

        self.mmap = self.mmap.rotate_to_angles([alpha, beta, gamma])
        

    # Plotting

    def plot_kin_img(self, ax, **kwargs):
        """
        Plots the kinematic image on the given axes.

        Parameters:
        - ax: The axes object on which to plot the image.
        - **kwargs: Additional keyword arguments to be passed to the `ax.imshow` function.

        Returns:
        - The modified axes object.

        Example usage:
        ```
        fig, ax = plt.subplots()
        frame.plot_kin_img(ax, cmap='gray')
        plt.show()
        ```
        """
        ax.imshow(self.kin_img, **kwargs)
        return ax
    
    def plot_kin_middle_axis(self, ax, label='labeled kin middle axis', delta=np.array([0,0]),**kwargs):
        """
        Plots the kinematic axis on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the kinematic axis.
            **kwargs: Additional keyword arguments to be passed to the `ax.plot` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        ax.plot(self.kin_middle_axis[:, 0]+delta[0], self.kin_middle_axis[:, 1]+delta[1], label=label, **kwargs)
        return ax
    
    def plot_kin_top_axis(self, ax, label='labeled kin top axis', delta=np.array([0,0]),**kwargs):
        """
        Plots the top kinematic axis on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the top kinematic axis.
            **kwargs: Additional keyword arguments to be passed to the `ax.plot` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        ax.plot(self.kin_top_axis[:, 0]+delta[0], self.kin_top_axis[:, 1]+delta[1], label=label, **kwargs)
        return ax

    def plot_map_axis_middle_view_on_kin(self, ax, label='map axis middle view',delta=(0,0), **kwargs):
        """
        Plots the projected map axis on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the map axis.
            **kwargs: Additional keyword arguments to be passed to the `plot` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        axis = self.mmap.get_axis_points()

        # remove z coordinate
        axis = axis[:, [0, 1]] + delta

        ax.plot(axis[:, 0], axis[:, 1], label=label, **kwargs)
        return ax
    
    def plot_map_axis_points_middle_view_on_kin(
            self, 
            ax, 
            label='map axis points middle view', 
            delta=(0,0), 
            color=np.array([1,0,0]),
            s=20,
            **kwargs):
        """
        Plots the projected map axis points on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the map axis points.
            **kwargs: Additional keyword arguments to be passed to the `scatter` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        axis = self.mmap.get_axis_points()

        # remove z coordinate
        axis = axis[:, [0, 1]] + delta

        ax.scatter(axis[:, 0], axis[:, 1], label=label, color=color, s=s,**kwargs)
        return ax

    def plot_map_axis_top_view_on_kin(self, ax, label='map axis top view', delta=(0,0), **kwargs):
        """
        Plots the projected map axis on the y-z plane on the given ax.
        Align the axis with the kin_top_axis.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the map axis.
            **kwargs: Additional keyword arguments to be passed to the `plot` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        axis = self.mmap.get_axis_points()

        axis = axis - axis[0]

        #remove y coordinate
        axis = axis[:, [0, 2]]

        axis = axis + self.kin_top_axis[0] + delta

        ax.plot(axis[:, 0], axis[:, 1], label=label, **kwargs)
        return ax
    
    def plot_kinematic_vector(self, ax, length=50, top_label='Top kinematic axis', middle_label='Middle kinematic axis', delta=(0,0), **kwargs):
        """
        Plot the kinematic vector on the given axes.

        Parameters:
        - ax: The matplotlib Axes object to plot on.
        - delta: Optional. The delta to add to the kinematic vector.
        - **kwargs: Additional keyword arguments to pass to the `plot` function.

        Returns:
        - The modified matplotlib Axes object.

        """
        vec = self.compute_kinematic_vector()
        vec_top = vec[[0,2]]
        vec_middle = vec[[0,1]]

        segment_top = np.array([self.kin_top_axis[0], self.kin_top_axis[0] + vec_top*length]) + delta
        segment_middle = np.array([self.kin_middle_axis[0], self.kin_middle_axis[0] + vec_middle*length]) + delta

        ax.plot(segment_top[:, 0], segment_top[:, 1], color='r', label=top_label, **kwargs)
        ax.plot(segment_middle[:, 0], segment_middle[:, 1], color='b', label=middle_label, **kwargs)

        return ax

    def plot_map_on_frame(self, ax, colors=None, **kwargs):
        """
        Plot the projected muscles on a frame.

        Parameters:
        - ax: The matplotlib Axes object to plot on.
        - colors: Optional. A list of colors to use for each muscle. If not provided, default colors will be generated.
        - **kwargs: Additional keyword arguments to pass to the `plot` function.

        Returns:
        - The modified matplotlib Axes object.

        """
        muscles = self.mmap.get_muscles()
        if colors is None:
            colors = mp.get_equally_spaced_colors(len(muscles))

        muscles_pts = self.compute_projected_muscle_points()
        for pts, c in zip(muscles_pts, colors):
            ax.scatter(pts[:, 0], pts[:, 1], color=c, **kwargs)
        return ax
    
    def plot_coordinate_frame(self, ax, colors=np.array([[1,0,0], [0,1,0]]), origin=[10,10], frame_size=100, **kwargs):
        """
        Plot a x-y coordinate frame on the given axes.

        Parameters:
        - ax: The axes object on which to plot the coordinate frame.
        - colors: An array of shape (2, 3) representing the colors of the x-axis and y-axis respectively.
        - origin: The coordinates of the origin of the coordinate frame.
        - frame_size: The size of the coordinate frame.
        - **kwargs: Additional keyword arguments to be passed to the `plot` function.

        Returns:
        - ax: The modified axes object with the coordinate frame plotted.
        """
        x_axis = np.array([[0,0],[1,0]]) * frame_size + origin
        y_axis = np.array([[0,0],[0,1]]) * frame_size + origin

        ax.plot(x_axis[:, 0], x_axis[:, 1], c=colors[0], label='x-axis, vector: [1,0]', **kwargs)
        ax.plot(y_axis[:, 0], y_axis[:, 1], c=colors[1], label='y-axis, vector: [0,1]', **kwargs)    

        return ax

    def plot_convex_hulls_on_middle_view(self, ax, colors=None, labels=None, **kwargs):
        """
        Plot the convex hulls of the muscles on the given axes.

        Parameters:
        - ax: The axes object on which to plot the convex hulls.
        - colors: Optional. A list of colors to use for each convex hull. If not provided, a set of equally spaced colors will be used.
        - labels: Optional. A list of labels for each convex hull. If provided, the labels will be displayed on the plot.
        - **kwargs: Additional keyword arguments to be passed to the `convex_hull_plot_2d` function.

        Returns:
        - The modified axes object.

        """
        if colors is None:
            colors = mp.get_equally_spaced_colors(len(self.mmap.get_muscles()))
        if labels is None:
            labels = [None] * len(self.mmap.get_muscles())
        hulls = self.compute_projected_hulls()
        for hull, c, label in zip(hulls, colors, labels):
            points = hull.points
            plt.plot(points[hull.vertices,0], points[hull.vertices,1], color=c, label=label, **kwargs)
        return ax

    def plot_muscle_img(self, ax, **kwargs):
        """
        Plot the muscle image on the given axes.

        Parameters:
        - ax: The axes object on which to plot the image.
        - **kwargs: Additional keyword arguments to be passed to the `imshow` function.

        Returns:
        - The modified axes object.

        """
        ax.imshow(self.muscle_img, **kwargs)
        return ax

    def plot_muscle_middle_axis(self, ax, label='labeled muscle middle axis', delta=np.array([0,0]), **kwargs):
        """
        Plot the muscle middle axis on the given axes.

        Parameters:
        - ax: The axes object on which to plot the muscle middle axis.
        - label: The label of the plotted line.
        - delta: Optional. The delta to add to the muscle middle axis.
        - **kwargs: Additional keyword arguments to be passed to the `plot` function.

        Returns:
        - The modified axes object.

        """
        ax.plot(self.muscle_middle_axis[:, 0]+delta[0], self.muscle_middle_axis[:, 1]+delta[1], label=label, **kwargs)
        return ax
    
    def plot_map_axis_middle_view_on_muscle(
            self, 
            ax, 
            label='map axis middle view', 
            delta=(0,0), 
            **kwargs):
        """
        Plot the projected map axis on the muscle image.

        Parameters:
        - ax: The axes object on which to plot the map axis.
        - label: The label of the plotted line.
        - delta: Optional. The delta to add to the map axis.
        - **kwargs: Additional keyword arguments to be passed to the `plot` function.

        Returns:
        - The modified axes object.

        """
        axis = self.mmap.get_axis_points()

        # remove z coordinate
        axis = axis[:, [0, 1]] + delta

        ax.plot(axis[:, 0], axis[:, 1], label=label, **kwargs)
        return ax

    # Getters
    def get_img(self):
        return self.kin_img  
    def get_kin_middle_axis(self):
        return self.kin_middle_axis_points
    def get_kin_top_axis(self):
        return self.kin_top_axis
    def get_mmap(self):
        return self.mmap
    def get_muscle_img(self):
        return self.muscle_img
    def get_muscle_middle_axis(self):
        return self.muscle_middle_axis

    # Setters
    def set_kin_middle_axis(self, axis):
        # assert axis 2d
        assert axis.shape == (2, 2), \
            f'Expected 2D array of shape (2, 2), got {axis.shape}'
        
        self.kin_middle_axis = axis
    def set_kin_top_axis(self, axis):
        # assert axis 2d
        assert axis.shape == (2, 2), \
            f'Expected 2D array of shape (2, 2), got {axis.shape}'
        self.kin_top_axis = axis
    def set_mmap(self, mmap):
        assert type(mmap) == mp.MuscleMap, \
            f'Expected a MuscleMap object, got {type(mmap)}'
        self.mmap = mmap
    def set_img(self, kin_img):
        assert type(kin_img) == np.ndarray, \
            f'Expected a numpy array, got {type(kin_img)}'
        self.kin_img = kin_img
    def set_muscle_img(self, muscle_img):
        if muscle_img is not None:
            assert type(muscle_img) == np.ndarray, \
            f'Expected a numpy array, got {type(muscle_img)}'
        self.muscle_img = muscle_img
    def set_muscle_middle_axis(self, muscle_middle_axis):
        if muscle_middle_axis is not None:
            assert muscle_middle_axis.shape == (2, 2), \
                f'Expected 2D array of shape (2, 2), got {muscle_middle_axis.shape}'
        self.muscle_middle_axis = muscle_middle_axis
    # Computers
    def compute_projected_muscle_points(self):
        points = []
        for muscle in self.mmap.get_muscles():
            pts = muscle.get_points()
            # remove z coordinate
            projected_pts = pts[:, :2]
            points.append(projected_pts)

        return points
    
    def compute_kin_middle_axis_vector(self, unit=True):
        """
        Compute the unit vector of the kinematic axis.

        Args:
        - unit (bool): Flag indicating whether to return the unit vector (default: True).

        Returns:
        - numpy.ndarray: The vector of the kinematic axis. If `unit` is True, the vector is normalized to unit length.
        """

        vec = self.kin_middle_axis[1] - self.kin_middle_axis[0]

        if unit:
            vec = vec / np.linalg.norm(vec)

        return vec
    
    def compute_kin_top_axis_vector(self, unit=True):
        """
        Compute the unit vector of the top kinematic axis.

        Args:
        - unit (bool): Flag indicating whether to return the unit vector (default: True).

        Returns:
        - numpy.ndarray: The vector of the top kinematic axis. If `unit` is True, the vector is normalized to unit length.
        """

        vec = self.kin_top_axis[1] - self.kin_top_axis[0]

        if unit:
            vec = vec / np.linalg.norm(vec)

        return vec
       
    def compute_kinematic_vector(self, unit=True):
        """
        Compute the kinematic vector derived from the middle and top views.

        Returns:
        - The kinematic vector.
        """

        l1 = self.compute_kin_middle_axis_vector(unit=False)
        l2 = self.compute_kin_top_axis_vector(unit=False)

        # kinematic vector l = (lx, ly, lz)

        l1x, l1y = l1[0], l1[1]
        l2x, l2z = l2[0], l2[1]

        lx = l1x # since mapping is for the ventral/middle view, it's better to be aligned with the x middle view
        ly = l1y
        lz = l2z

        l = np.array([lx, ly, lz])

        if unit:
            l = l / np.linalg.norm(l)

        return l

    def compute_muscle_middle_axis_vector(self, unit=True):
        """
        Compute the axis vector derived from the middle view of the muscle camera.

        Returns:
        - The vector.
        """
        vec = self.muscle_middle_axis[1] - self.muscle_middle_axis[0]

        if unit:
            vec = vec / np.linalg.norm(vec)

        return vec

    def compute_beta_from_kin_top_axis(self):
        """
        Compute the angle between the top kinematic axis and the x-axis (gamma/roll).

        Returns:
        - The angle between the kinematic axis and the x-axis.
        """
        y_axis_vec = np.array([0, 1]) # represent z-axis in 3D, the y-axis in 2D top view is the z-axis in 3D
        kin_top_axis_vec = self.compute_kin_top_axis_vector()
        angle = np.arccos(np.dot(y_axis_vec, kin_top_axis_vec))
        
        return angle

    def compute_kin_middle_axis_angle(self):
        """
        Compute the angle between the kinematic axis and the x-axis (alpha/yaw).

        Returns:
        - The angle between the kinematic axis and the x-axis.
        """
        x_axis_vec = np.array([1, 0])
        kin_middle_axis_vec = (self.kin_middle_axis[1] - self.kin_middle_axis[0])/ np.linalg.norm(self.kin_middle_axis[1] - self.kin_middle_axis[0])
        angle = np.arccos(np.dot(x_axis_vec, kin_middle_axis_vec))
        return angle
    
    def compute_kin_top_axis_angle(self, correct=True):
        """
        Compute the angle between the top kinematic axis and the x-axis (beta/pitch).

        Returns:
        - The angle between the kinematic axis and the x-axis.
        """
        x_axis_vec = np.array([1, 0])
        kin_top_axis_vec = self.compute_kin_top_axis_vector()
        angle = np.arccos(np.dot(x_axis_vec, kin_top_axis_vec))

        if kin_top_axis_vec[1] > 0 and correct:
            angle = angle + np.pi


        return angle
        
    def compute_kin_map_ratio(self):
        """
        Compute the ratio between the kinematic axis and the map axis.

        Returns:
        - The ratio between the kinematic axis and the map axis.
        """
        kinematic_vector = self.compute_kinematic_vector(unit=False)
        map_axis_vec = self.mmap.compute_axis_vector(unit=False)

        ratio = np.linalg.norm(kinematic_vector) / np.linalg.norm(map_axis_vec)

        return ratio

    def compute_length_muscle_vector(self):
        """
        Compute the length of the muscle vector.

        Returns:
        - The length of the muscle vector.
        """
        muscle_middle_axis_vec = self.compute_muscle_middle_axis_vector(unit=False)
        angle = self.compute_kin_top_axis_angle()

        l1 = np.linalg.norm(muscle_middle_axis_vec)

        l_tot = l1 / np.cos(angle) 
        # length of the muscle vector, derived with the kin top axis angle
        # l1 is the total length of the vector projected on the x-y plane
        #  and `angle` the angle between the z-axis and the x-axis
        # so l_tot * cos(angle) = l1

        return l_tot

    def compute_muscle_vector(self, unit=True):
        """
        Compute the muscle vector derived from the middle view of the muscle camera.

        Returns:
        - The muscle vector.
        """
        l1 = self.compute_muscle_middle_axis_vector(unit=False)
        angle = self.compute_kin_top_axis_angle()

        x, y = l1[0], l1[1]
    
        # right triangle, with `angle` and `l1` as the adjacent side
        z = np.linalg.norm(l1) * np.tan(angle)

        l = np.array([x, y, z])

        if unit:
            l = l / np.linalg.norm(l)

        return l        

    def compute_muscle_map_ratio(self):
        """
        Compute the ratio between the muscle axis and the kinematic axis.

        Returns:
        - The ratio between the muscle axis and the kinematic axis.
        """
        muscle_middle_axis_vec = self.compute_muscle_middle_axis_vector(unit=False)

        l1 = np.linalg.norm(muscle_middle_axis_vec)

        l_tot = self.compute_length_muscle_vector()
        
        map_axis_vec = self.mmap.compute_axis_vector(unit=False)

        ratio = l_tot/np.linalg.norm(map_axis_vec)

        return ratio

    def compute_projected_hulls(self):
        """
        Compute the projected convex hulls of the muscles.

        Returns:
        - The projected convex hulls of the muscles.
        """
        muscles_pts = self.compute_projected_muscle_points()
        hulls = []
        for pts in muscles_pts:
            hull = ConvexHull(pts)
            hulls.append(hull)
        return hulls
