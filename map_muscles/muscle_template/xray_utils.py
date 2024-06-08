from _root_path import add_root
add_root()

import pandas as pd
import json
import numpy as np

import map_muscles.path_utils as pu

"""
The main of this file extract the leg muscles from the muscle_state json file and save them in csv files.

This file also contains functions to preprocess the data frames of the leg muscles,
as well as getting a list of data frames of the femur muscles.
"""

data_path = pu.get_xray_dir()
muscle_state_path = data_path / 'muscles_state.json'
muscle_with_joints_state_path = data_path / 'muscles_with_joints_state.json'

def load_muscle_state(path=muscle_state_path):
    """
    Load the muscle state from the json file
    """
    with open(path, 'r') as f:
        muscles = json.load(f)
    return muscles

def get_leg(data_dir=data_path, leg='LH'):
    """
    Get the leg fibers
    """
    leg = pd.read_csv(data_dir / f'{leg}.csv')
    return leg

def get_femur_muscles(data_dir=data_path, leg='LH', femur_id='Fe', lines=True, remove=False, to_remove=['LH TrFe', 'LF TrFe', 'RM TrFe']):
    """
    Get the muscles associated with the femur.

    Args:
        data_dir (str): The directory path where the data is stored. Default is `data_path`.
        leg (str): The leg identifier. Default is 'LH'.
        femur_id (str): The identifier for the femur muscles. Default is 'Fe'.
        lines (bool): Whether to add lines to the muscle dataframes. Default is True.
        remove (bool): Whether to remove specific muscles from the result. Default is False.
        to_remove (list): List of muscle names to remove from the result. Default is ['LH TrFe'].

    Returns:
        list: A list of dataframes, each containing a muscle associated with the femur.
    """
    lh = get_leg(data_dir, leg)
    layers_names = lh['layer_name'].unique()
    fe_ids = [layer for layer in layers_names if femur_id in layer]
    femur_muscles = [lh[lh['layer_name'] == fe] for fe in fe_ids]

    if remove:
        femur_muscles = [muscle for muscle in femur_muscles if muscle['layer_name'].iloc[0] not in to_remove]

    if lines:
        femur_muscles = [add_lines_to_df(muscle) for muscle in femur_muscles]

    return femur_muscles

# function to create a line from two points
def create_line(p1, p2, str=True):
    """
    Create a line in 3D space between two points.

    Parameters:
        p1 (tuple): The coordinates of the first point (x, y, z).
        p2 (tuple): The coordinates of the second point (x, y, z).
        str (bool, optional): Indicates whether the input points are strings that need to be evaluated. Defaults to True.

    Returns:
        tuple: Three lists representing the x, y, and z coordinates of the line.
    """
    if str:
        p1 = eval(p1)
        p2 = eval(p2)
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    return x, y, z

def add_lines_to_df(df, A_key='pointA', B_key='pointB'):
    """
    Add a new column 'line' to the DataFrame `df`
    by evaluating the values of columns `A_key` and `B_key` as numpy arrays 
    and storing them as a single numpy array representing a line in the 'line' column.

    Parameters:
    - df: DataFrame
        The input DataFrame to which the 'line' column will be added.
    - A_key: str, optional
        The name of the column containing the values for point A. Default is 'pointA'.
    - B_key: str, optional
        The name of the column containing the values for point B. Default is 'pointB'.

    Returns:
    - df: DataFrame
        The input DataFrame with the new 'line' column added.

    Raises:
    - AssertionError: If the type of the elements in the 'line' column is not numpy.ndarray.
    """
    
    df['line'] = df.apply(lambda row: np.array([eval(row[A_key]), eval(row[B_key])]), axis=1)

    # assert type of element in line is np.array
    assert type(df['line'].iloc[0]) == np.ndarray
    
    return df

def isolate_leg(df, leg='LH'):
    """
    Isolate the leg fibers from the DataFrame `df` based on the leg identifier.

    Parameters:
    - df: DataFrame
        The input DataFrame containing the leg fibers.
    - leg: str, optional
        The leg identifier. Default is 'LH'.

    Returns:
    - leg: DataFrame
        The DataFrame containing the leg fibers of the specified leg.
    """
    leg = df[df['leg'] == leg]
    leg = leg.drop(columns='leg')
    return leg

def save_legs(
        muscle_state_path=muscle_state_path, 
        save_dir=data_path,
        legs_df_root_name=''
        ):
    """
    Save the leg fibers data frames in csv files.

    Parameters:
    - save_dir: str, optional
        The directory path where the data is stored. Default is `save_dir`.
    """
    muscles = load_muscle_state(muscle_state_path)
    anno_layers = [layer for layer in muscles['layers'] if layer['type'] == 'annotation']

    fiber_lines = []
    for layer in anno_layers:
        for anno in layer['annotations']:
            if anno['type'] != 'line':
                continue
            fiber_lines.append({
                #'id': anno['id'],
                'pointA': [int(i) for i in anno['pointA']],
                'pointB': [int(i) for i in anno['pointB']],
                'layer_name': layer['name'],
            })

    fiber_lines = pd.DataFrame(fiber_lines)
    # If layer name starts with LF, RF, LM, RM, LH, or RH, then it's a leg muscle
    fiber_lines['is_leg_muscle'] = fiber_lines['layer_name'].str.startswith(('LF', 'RF', 'LM', 'RM', 'LH', 'RH'))
    fiber_lines.loc[~fiber_lines.is_leg_muscle, 'leg'] = None
    leg_fibers = fiber_lines[fiber_lines.is_leg_muscle].copy()
    leg_fibers['leg'] = leg_fibers['layer_name'].str[0:2]
    leg_fibers['origin-insertion'] = leg_fibers['layer_name'].str.split(' ').str[1]
    leg_fibers['origin'] = leg_fibers['origin-insertion'].str[:2]
    leg_fibers['insertion'] = leg_fibers['origin-insertion'].str[2:]
    leg_fibers = leg_fibers.drop(columns='is_leg_muscle')
    
    lh = isolate_leg(leg_fibers,leg='LH')
    lf = isolate_leg(leg_fibers,leg='LF')
    rm = isolate_leg(leg_fibers,leg='RM')

    lh.to_csv(save_dir / f'LH{legs_df_root_name}.csv', index=False)
    lf.to_csv(save_dir / f'LF{legs_df_root_name}.csv', index=False)
    rm.to_csv(save_dir / f'RM{legs_df_root_name}.csv', index=False)

    print(f'Leg fibers data frames saved in {save_dir}')

def extract_femur_joint_layer(
        muscle_state_path=muscle_with_joints_state_path,
        save_dir=data_path,
        femur_joint_layer_name='Femur segment',
        femur_df_save_name = 'femur_joints.csv'
        ):
    
    state = load_muscle_state(muscle_state_path)
    layers = state['layers']

    femur_layer = [layer for layer in layers if layer['name'] == femur_joint_layer_name]
    femur_layer = femur_layer[0]

    femur_layer_df = pd.DataFrame(femur_layer['annotations'])

    femur_layer_df.to_csv(save_dir / femur_df_save_name, index=False)

    print(f'Femur joint layer saved in {save_dir}')

def get_lf_femur_joints(
    femur_joint_df_path=pu.get_xray_dir()/'femur_joints.csv',
    points_key='point',
    reverse_order=True
    ):
    
    femur = pd.read_csv(femur_joint_df_path)
    joints = femur[points_key].values
    pts = np.array([np.array(eval(joint)) for joint in joints])

    if reverse_order:
        pts = pts[::-1]
                   
    assert pts.shape == (2, 3), \
    f'Expected shape of joints array (2, 3), got {pts.shape}'

    return pts

if __name__ == "__main__":
    #save_legs()    

    save_legs() # only the femur layer was added, the fibers are the same as in muscle_state_path
    

    #extract_femur_joint_layer()



