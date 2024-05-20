from _root_path import add_root
add_root()

import numpy as np
import map_muscles.muscle_template.euler_map as mp

class MappedFrame():
    kin_img: np.ndarray
    kin_middle_axis: np.ndarray # 2D array of shape (2, 2) (middle view)
    kin_top_axis: np.ndarray # 2D array of shape (2, 2) (top view)
    mmap: mp.MuscleMap

    def __init__(
            self, 
            kin_img: np.ndarray, 
            kin_middle_axis:np.ndarray, 
            kin_top_axis:np.ndarray,
            mmap: mp.MuscleMap):
        self.set_img(kin_img)
        self.set_kin_middle_axis(kin_middle_axis)
        self.set_kin_top_axis(kin_top_axis)
        self.set_mmap(mmap)


    # Transformations
    def align_map_axis_ref_points(self):
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

    def scale_map(self, ratio=None):
        """
        Scales the map by a given ratio.
        Default: if ratio is None, compute the ratio between the kinematic axis and the map axis.

        Parameters:
            ratio (float): The ratio by which to scale the map.

        Returns:
            None
        """

        ref_point = self.mmap.get_axis_points()[0]

        if ratio is None:
            ratio = self.compute_axis_ratio()

        cmmap = self.mmap.centered_on_axis_point()

        smmap = cmmap.scale(ratio)

        self.mmap = smmap.translate(ref_point)
    
    def orient_map(self):
        """
        Orient the map to match the kinematic and top kinematic derived angles.

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
    
    def plot_kin_middle_axis(self, ax, **kwargs):
        """
        Plots the kinematic axis on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the kinematic axis.
            **kwargs: Additional keyword arguments to be passed to the `ax.plot` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        ax.plot(self.kin_middle_axis[:, 0], self.kin_middle_axis[:, 1], **kwargs)
        return ax
    
    def plot_kin_top_axis(self, ax, **kwargs):
        """
        Plots the top kinematic axis on the given axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the top kinematic axis.
            **kwargs: Additional keyword arguments to be passed to the `ax.plot` function.

        Returns:
            matplotlib.axes.Axes: The modified axes object.

        """
        ax.plot(self.kin_top_axis[:, 0], self.kin_top_axis[:, 1], **kwargs)
        return ax

    def plot_map_axis_middle_view(self, ax, delta=(0,0), **kwargs):
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

        ax.plot(axis[:, 0], axis[:, 1], **kwargs)
        return ax
    
    def plot_map_axis_top_view(self, ax, delta=(0,0), **kwargs):
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

        ax.plot(axis[:, 0], axis[:, 1], **kwargs)
        return ax

    def plot_kinematic_vector(self, ax, length=50, delta=(0,0), **kwargs):
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

        ax.plot(segment_top[:, 0], segment_top[:, 1], color='r', label='Top kinematic axis')
        ax.plot(segment_middle[:, 0], segment_middle[:, 1], color='b', label='Middle kinematic axis')

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


    # Getters
    def get_img(self):
        return self.kin_img  
    def get_kin_middle_axis(self):
        return self.kin_middle_axis_points
    def get_kin_top_axis(self):
        return self.kin_top_axis
    def get_mmap(self):
        return self.mmap

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

    # Computers

    def compute_projected_muscle_points(self):
        points = []
        for muscle in self.mmap.get_muscles():
            pts = muscle.get_points()
            # remove z coordinate
            projected_pts = pts[:, :2]
            points.append(projected_pts)

        return points
    
    def compute_axis_ratio(self):
        """
        Compute the ratio between the kinematic axis and the map axis.

        Returns:
        - The ratio between the kinematic axis and the map axis.
        """

        axis_pts = self.mmap.get_axis_points()
        map_axis_norm = np.linalg.norm(axis_pts[1] - axis_pts[0])
        kin_middle_axis_norm = np.linalg.norm(self.kin_middle_axis[1] - self.kin_middle_axis[0])

        return kin_middle_axis_norm / map_axis_norm

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
        

