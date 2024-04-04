from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import pandas as pd
import json
import numpy as np

data_path = Path(get_root_path()) / 'map_muscles' / 'data' / 'xray'
muscle_state_path = data_path / 'muscles_state.json'

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

def get_femur_muscles(data_dir=data_path, leg='LH', femur_id = 'Fe', segments=True, remove=False, to_remove=['LH TrFe']):
    lh = get_leg(data_dir, leg)
    layers_names = lh['layer_name'].unique()
    fe_ids = [layer for layer in layers_names if femur_id in layer]
    femur_muscles = [lh[lh['layer_name'] == fe] for fe in fe_ids]

    if remove:
        femur_muscles = [muscle for muscle in femur_muscles if muscle['layer_name'].iloc[0] not in to_remove]

    if segments:
        femur_muscles = [add_lines_to_df(muscle) for muscle in femur_muscles]

    return femur_muscles

def get_points(df, A_key='pointA', B_key='pointB'):
    """
    Get the points from the dataframe
    """

    A = df[A_key]
    B = df[B_key]
    points = np.concatenate((A, B), axis=0)

    # to array
    points = np.array([eval(point) for point in points])
    
    return points

# function to create a line from two points
def create_line(p1, p2, str = True):
    if str:
        p1 = eval(p1)
        p2 = eval(p2)
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    return x, y, z

def add_lines_to_df(df, A_key='pointA', B_key='pointB'):
    df['line'] = df.apply(lambda row: np.array([eval(row[A_key]), eval(row[B_key])]), axis=1)

    # assert type of element in line is np.array
    assert type(df['line'].iloc[0]) == np.ndarray
    
    return df


if __name__ == "__main__":
    np.random.seed(0)
    muscles = load_muscle_state()
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

    """
    leg = leg_fibers['leg']
    leg.describe()
    leg.unique()
    # array(['LF', 'RM', 'LH'], dtype=object)
    """

    def get_leg(df=leg_fibers, leg='LH'):
        leg = df[df['leg'] == leg]
        leg = leg.drop(columns='leg')
        return leg
    
    lh = get_leg(leg='LH')
    lf = get_leg(leg='LF')
    rm = get_leg(leg='RM')

    # save the legs
    lh.to_csv(data_path / 'LH.csv', index=False)
    lf.to_csv(data_path / 'LF.csv', index=False)
    rm.to_csv(data_path / 'RM.csv', index=False)



