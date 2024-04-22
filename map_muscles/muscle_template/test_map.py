from _root_path import add_root, get_root_path
add_root()

import map_muscles.muscle_template.map as mp
import map_muscles.path_utils as pu

import numpy as np
import open3d as o3d
from pathlib import Path

basic_muscle_name = "id_1_basic_d1.0_v1.0_LH_FeTi_flexor.pcd.npy"

# colors 
k = np.array([0,0,0])
r = np.array([1,0,0])
g = np.array([0,1,0])
b = np.array([0,0,1])
w = np.array([1,1,1])
reddish = np.array([1,0.5,0.5])

### Muscle tests ###

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
    
def test_get_axis_center():
    points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    m = mp.Muscle(points, "test")


    axis_points = np.array([[0,0,0],[1,1,1]])
    m.set_axis_points(axis_points)
    centre = [0.5,0.5,0.5]
    assert np.allclose(centre, m.get_axis_centre())

    axis_points = np.array([[0,0,0],[0,0,1]])
    m.set_axis_points(axis_points)
    centre = [0,0,0.5]
    assert np.allclose(centre, m.get_axis_centre())

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
points = muscle.points
com = np.mean(points, axis=0)
direction = np.array([0.5,0.5,-0.3])
axis_points = np.array([com-500*direction, com + 500*direction])
muscle.set_axis_points(axis_points)
axis_center = muscle.get_axis_centre()

frame_center = com - 600*direction - 200*np.array([1,1,1])
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=frame_center)

def test_visualization(m=muscle):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    m.draw_axis(vis)
    m.draw_points(vis, color=np.array([0,0,0]))

    frame_center = com - 600*direction - 200*np.array([1,1,1])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=frame_center)
    vis.add_geometry(frame)

    vis.run(); vis.destroy_window()
    
def test_visualize_rotation(m=muscle):
    rotvec = np.array([0,0,1])
    theta = np.pi/2

    m_rotated = m.rotate(rotvec, theta)

    vis = o3d.visualization.Visualizer()

    vis.create_window()
    
    
    m.draw_axis(vis, color=k)
    m.draw_points(vis, color=k)

    m_rotated.draw_axis(vis, color=reddish)
    m_rotated.draw_points(vis, color=reddish)

    rotaxis_points = [axis_center-500*rotvec, axis_center+500*rotvec]
    rotaxis_points = o3d.utility.Vector3dVector(rotaxis_points)
    rotaxis_lines = o3d.utility.Vector2iVector([[0,1]])
    rotaxis = o3d.geometry.LineSet(rotaxis_points, rotaxis_lines)

    rotaxis.paint_uniform_color(b)
    vis.add_geometry(rotaxis)

    vis.add_geometry(frame)

    vis.run(); vis.destroy_window()

def test_visualize_gradient_of_rotation(m=muscle, n=6):
    rotvec=np.array([0,0,1])
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    rotaxis_points = [axis_center-500*rotvec, axis_center+500*rotvec]
    rotaxis_points = o3d.utility.Vector3dVector(rotaxis_points)
    rotaxis_lines = o3d.utility.Vector2iVector([[0,1]])
    rotaxis = o3d.geometry.LineSet(rotaxis_points, rotaxis_lines)
    rotaxis.paint_uniform_color(g)
    vis.add_geometry(rotaxis)

    vis.add_geometry(frame)

    for i in range(n-1):
        theta = i*(2*np.pi)/n
        red = r/(n-1)*i
        blue = b/(n-1)*i        
        m_rotated = m.rotate(rotvec, theta)
        m_rotated.draw_axis(vis, color=blue)
        m_rotated.draw_points(vis, color=red)

    vis.run(); vis.destroy_window()


### MuscleMap tests ###

dir_path = pu.get_basic_map_dir()

def test_load_MuscleMap(path:str=dir_path):
    mmap = mp.MuscleMap.from_directory(path)

    assert mmap.name == "basic_map"

    assert mmap.axis_points == None
    assert mmap.axis_vector == None

    for muscle in mmap.muscles:
        assert_all_none(muscle)

    
def test_MuscleMap_draw_points(path:str=dir_path):
    mmap = mp.MuscleMap.from_directory(path)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # all blue
    colors = np.tile(b, (len(mmap.muscles),1))

    mmap.draw_points(vis, colors=colors)
    
    vis.run(); vis.destroy_window()

def test_MuscleMap_draw_axis(path:str=dir_path):
    mmap = mp.MuscleMap.from_directory(path)
    mmap.set_axis_points(axis_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_points(vis)
    mmap.draw_axis(vis)

    vis.run(); vis.destroy_window()

        
def test_MuscleMap_draw_default():
    mmap = mp.MuscleMap.from_directory(dir_path)
    mmap.set_axis_points(axis_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    mmap.draw_default(vis)

    vis.run(); vis.destroy_window()

def test_MuscleMap_visualize_rotate():
    mmap=mp.MuscleMap.from_directory(dir_path)
    mmap.set_axis_points(axis_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ks = np.tile(k, (len(mmap.muscles), 1))

    reddishs = np.tile(reddish, (len(mmap.muscles), 1))

    mmap.draw_points(vis, colors=ks)
    mmap.draw_axis(vis, color=k, length=1500)

    rotvec = np.array([0,0,1])

    mmap_rotated = mmap.rotate(rotvec, np.pi/2)

    mmap_rotated.draw_points(vis, colors=reddishs)
    mmap_rotated.draw_axis(vis, color=reddish)

    center = mmap.get_axis_centre()
    rotaxis = mp.generate_line(center, rotvec, length=1000)
    rotaxis.paint_uniform_color(g)

    vis.add_geometry(rotaxis)

    vis.run(); vis.destroy_window()

def test_MuscleMap_visualize_gradient_of_rotation(n=6):
    mmap = mp.MuscleMap.from_directory(dir_path)
    mmap.set_axis_points(axis_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()


    rotvec = np.array([0,0,1])
    rotaxis = mp.generate_line(axis_center, rotvec, length=1000)
    rotaxis.paint_uniform_color(g)
    vis.add_geometry(rotaxis)

    for i in range(n):
        theta = i*(2*np.pi)/n
        red = r/(n)*i
        reds = np.tile(red, (len(mmap.muscles), 1))

        mmap_rotated = mmap.rotate(rotvec, theta)
        mmap_rotated.draw_points(vis, colors=reds)
        mmap_rotated.draw_axis(vis, color=red)

    vis.run(); vis.destroy_window()

def test_MuscleMap_visualize_roll_by_rotate_on_axis(n=2):

    vis = o3d.visualization.Visualizer()
    vis.create_window()    

    mmap = mp.MuscleMap.from_directory(dir_path)
    mmap.set_axis_points(axis_points)

    rotvec = mmap.axis_vector
    rotaxis = mp.generate_line(axis_center, rotvec, length=2000)
    rotaxis.paint_uniform_color(g)
    vis.add_geometry(rotaxis)

    for i in range(n):
        theta = i*(2*np.pi)/n
        red = r/(n)*i
        reds = np.tile(red, (len(mmap.muscles), 1))

        mmap_rotated = mmap.rotate(rotvec, theta)
        mmap_rotated.draw_points(vis, colors=reds)
        mmap_rotated.draw_axis(vis, color=red, length=1000)

    vis.run(); vis.destroy_window()



    











    

if __name__ == "__main__":
    test_compute_yaw_and_compute_pitch()
    test_compute_axis_vector()
    test_Muscle_basic_constructor()
    test_get_axis_center()
    test_Muscle_from_array_file()
    
    test_visualization()

    test_visualize_rotation()

    test_visualize_gradient_of_rotation()

    test_load_MuscleMap()

    test_MuscleMap_draw_points()

    test_MuscleMap_draw_axis()

    test_MuscleMap_draw_default()

    test_MuscleMap_visualize_rotate()

    test_MuscleMap_visualize_gradient_of_rotation()

    test_MuscleMap_visualize_roll_by_rotate_on_axis()

    print("All tests passed.")
    


