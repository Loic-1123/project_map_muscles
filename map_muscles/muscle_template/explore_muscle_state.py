from _root_path import add_root, get_root_path
add_root()


"""
This file was used to download and explore the muscle state from the xray data and report the findings.
"""

import pandas as pd
from caveclient import CAVEclient
from pathlib import Path
import json
import os
client = CAVEclient()
auth = client.auth

import map_muscles.path_utils as pu

#auth.get_new_token()

client.auth.token = "d1dfc8e7d3ade48310f724fcbd62d34e"

#client.auth.save_token(client.auth.token)


# The state from https://github.com/NeLy-EPFL/neuromechfly-muscles-dev/issues/6#issuecomment-1881224757
muscles = client.state.get_state_json(5920404218576896)

# save the state to a file
save_path = pu.get_xray_dir() / 'muscles_state.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

print('Saving state to:', save_path)
with open(save_path, 'w') as f:
    json.dump(muscles, f)
print('State saved')



# exploring muscle state
type(muscles) # <class 'dict'>
muscles.keys() 
"""
dict_keys([
    'dimensions', 'position', 
    'crossSectionScale', 'projectionOrientation', 
    'projectionScale', 'layers', 'showSlices', 
    'gpuMemoryLimit', 'systemMemoryLimit', 
    'selectedLayer', 'layout', 'selection'
    ])
"""

muscles['dimensions'] # {'x': [1e-09, 'm'], 'y': [1e-09, 'm'], 'z': [1e-09, 'm']}
# probably voxel size in meters

muscles['position'] # [1492.5, 1261.3992919921875, 1933.39794921875]
# probably x,y,z coordinates, but meaning unclear

muscles['crossSectionScale'] # 0.4159911617512128

muscles['projectionOrientation'] # [-0.560043454170227, 0.3828834891319275, 0.42451366782188416, -0.599616289138794]
# meaning unclear

muscles['projectionScale'] # 7683.460822040686
# meaning unclear


#1111 layers 1111#
# layers
layers = muscles['layers']
type(layers) # <class 'list'>

layers_type = set([type(layer) for layer in layers])

type(muscles['layers'][0]) # <class 'dict'>

layers_keys = set([key for layer in layers for key in layer.keys()])
"""
{'source', 'segmentDefaultColor', 
'annotationColor', 'segments', 
'tab', 'archived', 
'name', 'objectAlpha', 
'tool', 'colorSeed', 'type', 'annotations'}
"""


#222 START layer types 222#
layer_types = set([layer['type'] for layer in layers])
"""
{'annotation', 'segmentation', 'image'}
"""
#222 END layer types 222#


#222 START layer sources 222#
layer_sources = [layer['source'] for layer in muscles['layers']]
layer_sources_types = set([type(layer['source']) for layer in muscles['layers']])
len(layer_sources) # 81
# dict, str

#333 START layer sources str 333#
layer_sources_str = [layer['source'] for layer in muscles['layers'] if type(layer['source']) == str]
"""
2: ['precomputed://gs://blanke-ramdya-flyct/PSI-synchrotron/Dmel_white_stitched-v1.jpeg.ng', 
'precomputed://gs://blanke-ramdya-flyct/PSI-synchrotron/Dmel_white_stitched-v1.jpeg.ng/meshes']
"""
#333 END layer str 333#

#333 START layer sources dict 333#
layer_sources_dict = [layer['source'] for layer in muscles['layers'] if type(layer['source']) == dict]
# len: 79

layer_sources_dict_keys = set([key for layer in layer_sources_dict for key in layer.keys()])
# {'url', 'transform'}

layer_sources_dict_urls = set([layer['url'] for layer in layer_sources_dict])
# only 1 url: {'local://annotations'}

layer_sources_dict_transforms = [layer['transform'] for layer in layer_sources_dict]
# list of x,y,z; typical value: {'outputDimensions': {'x': [1e-09, 'm'], 'y': [1e-09, 'm'], 'z': [1e-09, 'm']}}
#333 END layer sources dict 333#
#222 END layer sources 222#




#222 START layer tabs 222#
layer_tabs = [layer['tab'] for layer in muscles['layers']]

layer_tabs_types = set([type(layer['tab']) for layer in muscles['layers']])
# type: str

layer_tabs_unique = set(layer_tabs)
# {'source', 'rendering', 'annotations'}
#222 END layer tabs 222#

#222 START layer names 222#
layer_names = set([layer['name'] for layer in muscles['layers']])
len(layer_names) # 81
#222 END layer names 222#











#### annotation layers ####
anno_layers = [layer for layer in muscles['layers'] if layer['type'] == 'annotation']
df = pd.DataFrame(anno_layers)
df.info()
# remove 'type', 'source', 'archived'
df = df.drop(columns=['type', 'source', 'archived'])
df.describe()

# tool
tool = df['tool']
tool.describe()
tool.unique()
# array(['annotateLine', nan, 'annotatePoint'], dtype=object)

# nb 'annotateLine'
nb_annotateLine = tool.value_counts()['annotateLine']
# 76

# nb 'annotatePoint'
nb_annotatePoint = tool.value_counts()['annotatePoint']
# 2
point = df[df['tool'] == 'annotatePoint']

# nb nan
nb_nan = tool.isna().sum()
# 1

tabs= df['tab']
tabs.describe()
tabs.unique()
# array(['annotations', 'source'], dtype=object)

# nb 'annotations'
nb_annotations = tabs.value_counts()['annotations']
# 78

# nb 'source'
nb_source = tabs.value_counts()['source']
# 1


# annotations
annotations = df['annotations']
annotations.describe()
types = set([type(anno) for anno in annotations])
# list, list of ?

types = []
for anno in annotations:    
    for a in anno:
        types.append(type(a))

types = set(types)

anno = annotations[0]


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

# remove the is_leg_muscle column
leg_fibers = leg_fibers.drop(columns='is_leg_muscle')
leg_fibers.info()
leg_fibers.describe()

leg = leg_fibers['leg']
leg.describe()
leg.unique()
# array(['LF', 'RM', 'LH'], dtype=object)

orginin = leg_fibers['origin']
orginin.describe()
orginin.unique()
# array(['Th', 'Co', 'Tr', 'Fe', 'Ti', 'te'], dtype=object)

insertion = leg_fibers['insertion']
insertion.describe()
insertion.unique()
# array(['Co', 'Tr', 'Fe', 'Ti', 'Ta', 'Tr?', 'rgotrochanter', 'Cl', 'Co?'],

def get_leg(df=leg_fibers, leg='LH'):
    leg = df[df['leg'] == leg]
    leg = leg.drop(columns='leg')
    return leg

lh = get_leg(leg='LH')
lh.describe()

layer_name = lh['layer_name']
layer_name.describe()
layer_name.unique()

"""
array(['LH ThCo 1', 'LH ThCo 2', 'LH ThCo 3', 'LH ThCo? 4',
       'LH tergotrochanter', 'LH ThTr 2', 'LH ThTr 3 (same as 2?)',
       'LH ThTr 4', 'LH ThTr 5 (same as 4?)', 'LH CoTr 1', 'LH CoTr 2',
       'LH CoTr 3', 'LH CoTr 4', 'LH CoTr 5', 'LH TrFe', 'LH FeTi flexor',
       'LH FeTi anterior acc flexor', 'LH FeTi posterior acc flexor',
       'LH FeTi extensor', 'LH FeCl ltm2', 'LH TiCl ltm1',
       'LH TiTa 1 (detached origin)', 'LH TiTa 2',
       'LH TiTa 3 (detached origin)'], dtype=object)
"""

thco1 = lh[lh['layer_name'] == 'LH ThCo 1']
thco1.describe()

lf = get_leg(leg='LF')
lf.describe()

layer_name = lf['layer_name']
layer_name.describe()
layer_name.unique()

"""
array(['LF ThCo 1', 'LF ThCo 3 (same muscle as 1?)',
       'LF ThCo sternal anterior rotator', 'LF ThCo 4', 'LF ThCo 5',
       'LF ThCo 6 (same as 5?)', 'LF ThCo 7', 'LF ThTr tergotrochanter',
       'LF ThTr sternotrochanter', 'LF CoTr flexor?',
       'LF CoTr acc flexor?', 'LF CoTr extensor?',
       'LF CoTr acc extensor?', 'LF TrFe', 'LF FeTi flexor',
       'LF FeTi medial acc flexor', 'LF FeTi lateral acc flexor',
       'LF FeTi extensor', 'LF FeTa ltm2', 'LF TiTa 1', 'LF TiTa 2',
       'LF TiTa 3 (same as 1?)'], dtype=object)
"""

rm = get_leg(leg='RM')
rm.describe()

layer_name = rm['layer_name']
layer_name.describe()
layer_name.unique()    

"""
array(['RM ThCo 1', 'RM ThCo 2 (same as 1?)', 'RM ThCo 3',
       'RM ThCo 4 (same as 3?)', 'RM ThTr? 1', 'RM ThTr? 2 (insert=1)',
       'RM ThTr 3 (origin~=2, insert=1&2)', 'RM tergotrochanter',
       'RM ThTr? 4 (insert=tergotroch)', 'RM ThTr? 5 (insert=tergotroch)',
       'RM CoTr 1', 'RM CoTr 2 (origin ~= 1, insert=1)', 'RM CoTr 3',
       'RM TrFe', 'RM FeTi flexor', 'RM FeTi anterior acc flexor',
       'RM FeTi posterior acc flexor', 'RM FeTi extensor',
       'RM FeCl ltm2 (detached origin)', 'RM TiCl ltm1', 'RM TiTa 1',
       'RM TiTa 2', 'RM TiTa 3', 'RM TiTa 4'], dtype=object)
"""








