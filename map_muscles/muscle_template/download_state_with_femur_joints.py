from _root_path import add_root
add_root()

import pandas as pd
from caveclient import CAVEclient
from pathlib import Path
import json
import os

import map_muscles.path_utils as pu

"""
This is used to download the state with the femur joints labeled from the xray data.
"""

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





