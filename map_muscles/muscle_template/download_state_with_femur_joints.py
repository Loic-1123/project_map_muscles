from _root_path import add_root
add_root()

import pandas as pd
import numpy as np
#from caveclient import CAVEclient
from pathlib import Path
import json
import os

import map_muscles.path_utils as pu

"""
This file is used to download the state with the femur joints labeled from the xray data,
and test the joints positions.

"""


def download_state_with_femur_joints():

    client = CAVEclient()
    auth = client.auth

    #auth.get_new_token()

    client.auth.token = "d1dfc8e7d3ade48310f724fcbd62d34e"

    #client.auth.save_token(client.auth.token)

    state_id = 6586582729490432

    state = client.state.get_state_json(state_id)

    layers = state['layers']

    layers_keys = set([key for layer in layers for key in layer.keys()])
    #{'colorSeed', 'source', 'meshSilhouetteRendering', 'segmentDefaultColor', 'tab', 'annotationColor', 'type', 'segments', 'tool', 'objectAlpha', 'archived', 'annotations', 'name'}

    layer_names = set([layer['name'] for layer in state['layers']])
    len(layer_names) # 84

    # save state to file
    save_path = pu.get_xray_dir() / 'muscles_with_joints_state.json'
    print('Saving state to:', save_path)
    with open(save_path, 'w') as f:
        json.dump(state, f)
    print('State saved')

    # if layer['name'] is 'Femur segment'
    femur_layer = [layer for layer in state['layers'] if layer['name'] == 'Femur segment']

def joints_df_to_np_array(
    joints_df: pd.DataFrame,
    points_key:str = 'point',
    ):

    joints = joints_df[points_key].values

    # convert str '[x,y,z]' to np.array([x,y,z])

    joints = np.array([np.array(eval(joint)) for joint in joints])

    return joints



if __name__ == "__main__":
    #download_state_with_femur_joints()
    #xu.extract_femur_joint_layer()

    # check which leg correspond to the labeling of the femur joints
    import open3d as o3d
    import map_muscles.muscle_template.euler_map as mp


    femur_df_name = 'femur_joints.csv'
    femur = pd.read_csv(pu.get_xray_dir() / femur_df_name)
    femur.info()
    pts = joints_df_to_np_array(femur)

    

    # rm leg
    dir_path = pu.get_map_dir() / 'rm_leg_map'
    rmmmap = mp.MuscleMap.from_directory(dir_path)
    rmmmap.set_axis_points(pts)

    # lh leg
    dir_path = pu.get_basic_map_dir()
    lhmmap = mp.MuscleMap.from_directory(dir_path)
    lhmmap.set_axis_points(pts)

    # lf leg 
    dir_path = pu.get_map_dir() / 'lf_leg_map'
    lfmmap = mp.MuscleMap.from_directory(dir_path)
    lfmmap.set_axis_points(pts)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    n = len(rmmmap.get_muscles())
    rmcolors = [np.array([1, 0, 0]) for i in range(n)]
    rmmmap.draw_points(vis, colors = rmcolors)

    lhcolors = [np.array([0, 1, 0]) for i in range(n)] 
    lhmmap.draw_points(vis, colors = lhcolors)

    lfcolors = [np.array([0, 0, 1]) for i in range(n)]
    lfmmap.draw_points(vis, colors = lfcolors)

    rmmmap.draw_axis_points(vis)
    lhmmap.draw_axis(vis)
    lfmmap.draw_axis(vis)

    vis.run(); vis.destroy_window()

    vis.run(); vis.destroy_window()
    
    # only lf leg

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    lfmmap.draw_default(vis)

    vis.run(); vis.destroy_window()






    
    


    
    
    



