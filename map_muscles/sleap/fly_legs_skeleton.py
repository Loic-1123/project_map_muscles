# need another environment to run sleap: because of the following error:
"""
LibMambaUnsatisfiableError: Encountered problems while solving:
  - package sleap-1.3.3-py37_0 requires python 3.7.12, but none of the providers can be installed

Could not solve for environment specs
The following packages are incompatible
├─ pin-1 is installable and it requires
│  └─ python 3.12.* , which can be installed;
└─ sleap 1.3.3**  is not installable because it requires
   └─ python 3.7.12 , which conflicts with any installable versions previously reported.
"""
from _root_path import add_root, get_root_path
add_root()
from sleap import skeleton as sk
from pathlib import Path

# position 1: r (right), l (left)
# position 2: f (front), m (middle), b (back)

leg = 'leg'

trochanter = 'trochanter'

femur = 'femur'

tibia = 'tibia'

tarsus = 'tarsus'

leg_tip = 'tip'

# joints: trochanter, femur-tibia, tibia-tarsus

leg_joints = (
    trochanter,
    femur + '-' + tibia,
    tibia + '-' + tarsus,
    leg_tip
)

def create_legs_skeleton(skeleton_name:str,
                    leg_joints=leg_joints,
                    front_rear_positions = ['front','middle','back'],
                    sep='_'):
    """
    Create a legs skeleton object.

    Args:
        skeleton_name (str): The name of the skeleton.
        leg_joints: The list of leg joints.
        front_rear_positions (list, optional): The list of front and rear positions. Defaults to ['f','m','b'].
        image_position (list, optional): The list of image positions. Defaults to ['m'].
        sep (str, optional): The separator for joint names. Defaults to '_'.

    Returns:
        Skeleton: The leg skeleton object.
    """
    skeleton = sk.Skeleton(name=skeleton_name)

    # anchor nodes: head, thorax, abdomen
    anchor_nodes = ['head', 'thorax', 'abdomen']
    for node in anchor_nodes:
        skeleton.add_node(node)


    #legs

    for position in front_rear_positions:
        r_joints=[]
        l_joints=[]
        for joint in leg_joints:
            # create left and right joints
            r_joint = 'R' + position + sep + joint
            l_joint = 'L' + position + sep + joint

            r_joints.append(r_joint)
            l_joints.append(l_joint)

            skeleton.add_node(r_joint)
            skeleton.add_node(l_joint)
            # add symmetry
            skeleton.add_symmetry(r_joint, l_joint)

        # create edges between joints of the same leg
        for i in range(len(r_joints)-1):
            joint = r_joints[i]
            next_joint = r_joints[i+1]
            skeleton.add_edge(joint, next_joint)

            joint = l_joints[i]
            next_joint = l_joints[i+1]
            skeleton.add_edge(joint, next_joint)

    

    return skeleton

if __name__ == "__main__":

    root_path = Path(get_root_path())
    output_folder = root_path / 'map_muscles' / 'sleap' / 'skeletons'
    output_folder.mkdir(exist_ok=True)

    legs_skeleton = create_legs_skeleton('fly_legs')
    output_path = output_folder / 'fly_legs_skeleton.json'
    legs_skeleton.save_json(output_path)

    print('Skeleton saved to', output_path)

    



