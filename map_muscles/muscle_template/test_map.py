from _root_path import add_root, get_root_path
add_root()

import map_muscles.muscle_template.map as mp
import map_muscles.path_utils as pu

import numpy as np
import open3d as o3d
from pathlib import Path

basic_muscle_name = "id_1_basic_d1.0_v1.0_LH_FeTi_flexor.pcd.npy"

def test_compute_yaw_and_compute_pitch():
    v = np.array([1,0,0])

    yaw = mp.compute_yaw(v)
    pitch = mp.compute_pitch(v)

    assert np.isclose(yaw, 0), \
    f"Yaw of vector [1,0,0] should be 0, but got {yaw}."
    assert np.isclose(pitch, 0), \
    f"Pitch of vector [1,0,0] should be 0, but got {pitch}."


    v = np.array([0,1,0])

    yaw = mp.compute_yaw(v)
    pitch = mp.compute_pitch(v)

    assert np.isclose(yaw, np.pi/2), \
    f"Yaw of vector [0,1,0] should be pi/2, but got {yaw}."
    assert np.isclose(pitch, 0), \
    f"Pitch of vector [0,1,0] should be 0, but got {pitch}."

    v = np.array([0,0,1])

    yaw = mp.compute_yaw(v)
    pitch = mp.compute_pitch(v)

    assert np.isclose(yaw, 0),\
    f"Yaw of vector [0,0,1] should be 0, but got {yaw}."
    assert np.isclose(pitch, np.pi/2),\
    f"Pitch of vector [0,0,1] should be pi/2, but got {pitch}."

    v = np.array([1,1,1])

    yaw = mp.compute_yaw(v)
    pitch = mp.compute_pitch(v)

    assert np.isclose(yaw, np.pi/4),\
    f"Yaw of vector [1,1,1] should be pi/4, but got {yaw}."
    assert np.isclose(pitch, np.pi/4),\
    f"Pitch of vector [1,1,1] should be pi/4, but got {pitch}."

def test_compute_axis_vector():
    m = mp.Muscle(np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]), "test")

    def assert_vec(axis, vec_truth):
        m.set_axis_points(axis)
        axis_vector = m.compute_axis_vector()
        assert np.allclose(axis_vector, vec_truth), \
        f"Axis vector should be {vec_truth}, but got {axis_vector}."

    assert_vec(
        axis=np.array([[0,0,0], [1,0,0]]),
        vec_truth=np.array([1,0,0])
        )

    assert_vec(
        axis=np.array([[0,0,0], [0,1,0]]),
        vec_truth=np.array([0,1,0])
        )
    
    assert_vec(
        axis=np.array([[0,0,0], [0,0,1]]),
        vec_truth=np.array([0,0,1])
        )
    
    assert_vec(
        axis=np.array([[0,0,0], [1,1,1]]),
        vec_truth=np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
        )
    
def assert_axis_points_close(m:mp.Muscle, axis:np.ndarray):
    assert np.allclose(m.axis_points, axis),\
    f"Axis points should be {axis}, but got {m.axis_points}."

def assert_axis_vector_close(m:mp.Muscle, v:np.ndarray):
    assert np.allclose(m.axis_vector, v),\
    f"Axis vector should be {v}, but got {m.axis_vector}."

def assert_roll_none(m:mp.Muscle):
    assert m.roll == None,\
    f"Roll should be None, but got {m.roll}."

def assert_roll_close(m:mp.Muscle, roll:float):
    assert m.roll == roll,\
    f"Roll should be {roll}, but got {m.roll}."

def assert_yaw_pitch_none(m:mp.Muscle):
    assert m.yaw == None,\
    f"Yaw should be None, but got {m.yaw}."
    assert m.pitch == None,\
    f"Pitch should be None, but got {m.pitch}."

def assert_yaw_pitch_close(m:mp.Muscle, yaw:float, pitch:float):
    assert m.yaw == yaw,\
    f"Yaw should be {yaw}, but got {m.yaw}."
    assert m.pitch == pitch,\
    f"Pitch should be {pitch}, but got {m.pitch}."

def assert_axis_none(m:mp.Muscle):
    assert m.axis_points == None,\
    f"Axis points should be None, but got {m.axis_points}."
    assert m.axis_vector == None,\
    f"Axis vector should be None, but got {m.axis_vector}."

def assert_all_none(m:mp.Muscle):
    assert_axis_none(m)
    assert_roll_none(m)
    assert_yaw_pitch_none(m)

def assert_name(m:mp.Muscle, name:str):
    assert m.name == name,\
    f"Name should be {name}, but got {m.name}."

def test_Muscle_basic_constructor():
    axis = np.array([[0,0,0],[1,1,1]])
    points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])

    m = mp.Muscle(points, "test", axis)

    assert_axis_points_close(m, axis)

    assert_axis_vector_close(m, np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)]))
    
    assert_roll_none(m)

    assert_name(m, "test")

    m = mp.Muscle(points, "test", axis, np.pi/4)

    assert_roll_close(m, np.pi/4)

    assert_yaw_pitch_close(m, np.pi/4, np.pi/4)

    m = mp.Muscle(points, "test")

    assert_axis_none(m)
    assert_roll_none(m)
    assert_yaw_pitch_none(m)
    

def test_Muscle_from_array_file(
    file_name:str=basic_muscle_name,
    name_truth = "id_1_basic_d1.0_v1.0_LH_FeTi_flexor.pcd",
    dir_rel_path:str=pu.get_basic_map_dir()
    ):

    file_path = Path(dir_rel_path) / file_name
    assert file_path.exists(), \
        f"test_Muscle_from_array_file: {file_path} does not exist to try the test."
    
    m = mp.Muscle.from_array_file(file_path)

    assert_name(m, name_truth)

    assert_all_none(m)

    m = mp.Muscle.from_array_file(file_path, "test")

    assert_name(m, "test")

    assert_all_none(m)

muscle = mp.Muscle.from_array_file(pu.get_basic_map_dir()/basic_muscle_name)


def test_visualization(m=muscle):
    points = m.points
    # compute center of mass
    com = np.mean(points, axis=0)

    # compute axis points
    direction = np.array([0.5,0.5,-0.3])
    axis_points = np.array([com-500*direction, com + 500*direction])

    m.set_axis_points(axis_points)

    vis = o3d.visualization.Visualizer()

    vis.create_window()
    m.draw_axis(vis)
    m.draw_points(vis, color=np.array([0,0,0]))

    frame_center = com - 600*direction - 200*np.array([1,1,1])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=frame_center)
    vis.add_geometry(frame)

    vis.run(); vis.destroy_window()
    

def test_rotation(m=muscle):
    
    

    
    
    

def test_visualize_rotation():
    # TODO
    pass

if __name__ == "__main__":
    #test_compute_yaw_and_compute_pitch()
    #test_compute_axis_vector()
    #test_Muscle_basic_constructor()
    #test_Muscle_from_array_file()
    
    test_visualization()

    print("All tests passed.")
    


