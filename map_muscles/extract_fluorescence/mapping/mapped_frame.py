from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt


import map_muscles.path_utils as pu
import map_muscles.muscle_template.map as mp
import map_muscles.extract_fluorescence.plot_map_on_pixel_img as pmp

class MappedFrame():
    img: np.ndarray
    frame_axis: np.ndarray # 2D array of shape (2, 2)
    mmap: mp.MuscleMap
    

    def __init__(self, img: np.ndarray, img_axis:np.ndarray, mmap: mp.MuscleMap):
        self.set_img(img)
        self.set_img_axis(img_axis)
        self.set_mmap(mmap)


    # Plotting

    def plot_img(self, ax, **kwargs):
        ax.imshow(self.img, **kwargs)
        return ax
    
    def plot_img_axis(self, ax, **kwargs):
        ax.plot(self.frame_axis[:, 0], self.frame_axis[:, 1], **kwargs)
        return ax
    
    def plot_map_axis(self, ax, **kwargs):
        map2d = self.mmap.get_map2d()
        map2d.plot_axis(ax, **kwargs)
        return ax
    
    def plot_map_on_frame(self, ax, **kwargs):
        map2d = self.mmap.get_map2d()
        map2d.plot_maps(ax, **kwargs)
        return ax
    
    def plot_coordinate_frame(self, ax, colors=np.array([[1,0,0], [0,1,0]]), origin= [10,10], frame_size=100, **kwargs):
        x_axis = np.array([[0,0],[1,0]])*frame_size + origin
        y_axis = np.array([[0,0],[0,1]])*frame_size + origin

        ax.plot(x_axis[:, 0], x_axis[:, 1], c=colors[0], label= 'x-axis, vector: [1,0]', **kwargs)
        ax.plot(y_axis[:, 0], y_axis[:, 1], c=colors[1], label= 'y-axis, vector: [0,1]', **kwargs)    

        return ax

    # Getters
    def get_img(self):
        return self.img
    
    def get_img_axis(self):
        return self.frame_axis_points
    
    def get_mmap(self):
        return self.mmap
    
    

    # Setters
    def set_img_axis(self, img_axis):
        self.img_axis = img_axis

    def set_mmap(self, mmap):
        self.mmap = mmap

    def set_img(self, img):
        self.img = img

    
    
    

    



        
    

