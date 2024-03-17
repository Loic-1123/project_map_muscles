from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import pandas as pd
import json

data_path = Path(get_root_path()) / 'map_muscles' / 'data' / 'Xray'
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


# function to create a line from two points
def create_line(p1, p2, str = True):
    if str:
        p1 = eval(p1)
        p2 = eval(p2)
    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]
    z = [p1[2], p2[2]]
    return x, y, z





if __name__ == "__main__":
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



