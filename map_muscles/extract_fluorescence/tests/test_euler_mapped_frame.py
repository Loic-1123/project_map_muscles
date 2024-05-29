from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

import map_muscles.path_utils as pu
import map_muscles.extract_fluorescence.mapping.euler_mapped_frame as mf
import map_muscles.muscle_template.euler_map as mp
import map_muscles.extract_fluorescence.imaging_utils as imu


def get_locations_from_sleap_file(
        sleap_file_path,
        trochanter_id=0,
        femur_tibia_id=1,
        tibia_tarsus_id=2,
        top_trochanter_id=3,
        top_femur_tibia_id=4,
        ):
    with h5py.File(sleap_file_path, 'r') as f:
        locations = f["tracks"][:].T

    trochanter_locations = locations[:, trochanter_id, :, 0]
    femur_tibia_locations = locations[:, femur_tibia_id, :, 0]
    tibia_tarsus_locations = locations[:, tibia_tarsus_id, :, 0]
    top_trochanter_locations = locations[:, top_trochanter_id, :, 0]
    top_femur_tibia_locations = locations[:, top_femur_tibia_id, :, 0]

    return trochanter_locations, femur_tibia_locations, tibia_tarsus_locations, top_trochanter_locations, top_femur_tibia_locations

def get_labeled_kin_frame(
        npy_file_path,
        idx):
    kin_frames = np.load(npy_file_path)

    labeled_kin_frame = kin_frames[idx]

    return labeled_kin_frame

sleap_dir = pu.get_sleap_dir()
sleap_file = 'label_femur_V2.000_900_1440_kin.analysis.h5'
sleap_file_path = sleap_dir / sleap_file

tro, ft, tt, top_tro, top_ft = get_locations_from_sleap_file(sleap_file_path)

idx = len(tro) - 80

KINAXIS = np.array([tro[idx], ft[idx]])
TOPKINAXIS = np.array([top_tro[idx], top_ft[idx]])
kin_dir = pu.get_kin_dir()
kin_min = imu.get_min_id(kin_dir, 'jpg')

kin_frames_dir = pu.get_kin_frames_dir()
kin_file = 'kin_frames_3601_5800.npy'
kin_file_path = kin_frames_dir/ kin_file

KINFRAME = get_labeled_kin_frame(kin_file_path, idx)

map_dir = pu.get_basic_map_dir()
MMAP = mp.MuscleMap.from_directory(map_dir)
MMAP.init_default_axis_points()

FIGSIZE = (10, 10)

muscle_dir = pu.get_muscle_dir()
muscle_min = imu.get_min_id(muscle_dir, 'tif', id_format='{:06d}')
# muscle id = 1086 -> 186th frame


def get_mframe():
    mframe = mf.MappedFrame(KINFRAME, KINAXIS, TOPKINAXIS, MMAP)
    return mframe


def test_plot_coordinate_frame():
    mframe = get_mframe()

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_kin_img(ax, cmap='gray')
    mframe.plot_coordinate_frame(ax)

    ax.axis('off')
    ax.legend()

    plt.show()

def test_visualize_map_aligned_with_kin_middle_axis():
    """
    Test align_map_axis_ref_point_on_kin()
    """

    mframe = get_mframe()

    mframe.align_map_axis_ref_point_on_kin()
    
    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_kin_img(ax, cmap='gray')
    mframe.plot_kin_middle_axis(ax)
    mframe.plot_map_axis_middle_view_on_kin(ax)
    mframe.plot_map_on_frame(ax)

    ax.axis('off')
    ax.legend()

    plt.show()

def test_visualize_scaled_translated_map():
    """
    Test get_scaled_translated_map()
    """
    mframe = get_mframe()
    
    
    mframe.scale_map()
    mframe.align_map_axis_ref_point_on_kin()
    

    
    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_kin_img(ax, cmap='gray')
    mframe.plot_kin_middle_axis(ax)
    mframe.plot_map_axis_middle_view_on_kin(ax)
    mframe.plot_map_on_frame(ax, s=1)

    ax.axis('off')
    ax.legend()

    plt.show()

def angle_between(v1, v2):
    u1 = v1/np.linalg.norm(v1)
    u2 = v2/np.linalg.norm(v2)
    return np.arccos(np.dot(u1, u2))

def assert_same_angle(expected, angle, str=None, tol=1e-5):
    assert np.isclose(expected, angle, atol=tol), \
        f"Angles are not the same: expected: {expected}, got: {angle}. {str}"

def test_compute_kin_middle_axis_angle():
    mframe = get_mframe()
    x = np.array([1, 0])

    def test_kin_middle_axis(kin_middle_axis):
        kin_vec = kin_middle_axis[1] - kin_middle_axis[0]
        angle = angle_between(x, kin_vec)
        mframe.set_kin_middle_axis(kin_middle_axis)
        assert_same_angle(angle, mframe.compute_kin_middle_axis_angle(), str="Kin axis angle")
    
    n = 10
    
    angles = np.linspace(0, 2*np.pi, n)
    
    for angle in angles:
        vec = np.array([np.cos(angle), np.sin(angle)])
        test_kin_middle_axis(np.array([[0, 0], vec]))
            
def test_compute_kin_top_axis_angle():
    mframe = get_mframe()
    x = np.array([1, 0])

    def test_kin_top_axis(kin_top_axis):
        kin_vec = kin_top_axis[1] - kin_top_axis[0]
        # minus sign as top view is inverted
        angle = angle_between(x, kin_vec)
        mframe.set_kin_top_axis(kin_top_axis)
        assert_same_angle(angle, mframe.compute_kin_top_axis_angle(correct=False), str="Kin top axis angle")
    
    n = 10
    
    angles = np.linspace(0, 2*np.pi, n)
    
    for angle in angles:
        vec = np.array([np.cos(angle), np.sin(angle)])
        test_kin_top_axis(np.array([[0, 0], vec]))

def test_compute_kinematic_vector():
    mframe = get_mframe()

    mframe.align_map_axis_ref_point_on_kin()

    kin_vec = mframe.compute_kinematic_vector(unit=False)
    kin_top = mframe.compute_kin_top_axis_vector(unit=False)
    kin_middle = mframe.compute_kin_middle_axis_vector(unit=False)

    # assert kin_vec y = kin_middle_axis_vec y
    assert np.isclose(kin_vec[1], kin_middle[1]), \
        f"Kinematic vector y component is not the same as kin top axis y component: computed kin vec {kin_vec}, kin_middle: {kin_middle}"
    
    # assert kin_vec z = kin_top_axis_vec[1]
    assert np.isclose(kin_vec[2], kin_top[1]), \
        f"Kinematic vector z component is not the same as kin top axis y component: computed kin vec {kin_vec}, kin_top_vec: {kin_top}"
                      
    # assert kin_vec x = middle view x

    assert np.isclose(kin_vec[0], kin_middle[0]), \
        f"Kinematic vector x component is not the same as the mean of kin top axis x component and kin middle axis x component: computed kin vec {kin_vec}, kin_top_vec: {mframe.compute_kin_top_axis_vector()}, kin_middle_vec: {mframe.compute_kin_middle_axis_vector()}"

def test_visualize_compute_kinematic_vector():
    mframe = get_mframe()

    mframe.align_map_axis_ref_point_on_kin()

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    mframe.plot_kin_img(ax, cmap='gray')
    mframe.plot_kin_middle_axis(ax)
    mframe.plot_kin_top_axis(ax)
    mframe.plot_kinematic_vector(ax)
    
    ax.axis('off')
    ax.legend()

    plt.show()


def test_visualize_orient_map_on_kin():
    mframe = get_mframe()

    mframe.align_map_axis_ref_point_on_kin()
    mframe.orient_map_on_kin()

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    delta = np.array([10, 0])
    mframe.plot_kin_img(ax, cmap='gray')
    
    mframe.plot_kin_middle_axis(ax)
    mframe.plot_kin_top_axis(ax)
    
    mframe.plot_map_axis_top_view_on_kin(ax, delta=delta, label='Map axis top view')
    mframe.plot_map_axis_middle_view_on_kin(ax, delta=delta, label='Map axis middle view')

    mframe.plot_kinematic_vector(ax, delta=2*delta)

    ax.axis('off')
    ax.legend()

    plt.show()



def test_visualize_scaling_on_kinetic_frame():
    mframe = get_mframe()

    mframe.align_map_axis_ref_point_on_kin()
    mframe.orient_map_on_kin()
    ratio = mframe.compute_kin_map_ratio()
    mframe.scale_map(ratio=ratio)

    fig, ax = plt.subplots(1,1, figsize=FIGSIZE)

    delta = np.array([10, 0])
    mframe.plot_kin_img(ax, cmap='gray')
    
    mframe.plot_kin_middle_axis(ax, delta=2*delta)
    
    mframe.plot_convex_hulls_on_middle_view(ax)
    mframe.plot_map_axis_points_middle_view_on_kin(ax)

    ax.axis('off')
    ax.legend()

    plt.show()

def test_visualize_muscle_img():
    muscle_img = imu.get_matching_muscle_img(
        muscle_path=muscle_dir,
        kin_frame_id=idx+kin_min,
        min_id_kin=kin_min,
        kin_to_muscle_div_factor=4,
        min_id_muscle=muscle_min,
        id_format='{:06d}',
    )

    muscle_img = cv2.rotate(muscle_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    muscle_img = cv2.flip(muscle_img, 1)

    mframe = mf.MappedFrame(KINFRAME, KINAXIS, TOPKINAXIS, MMAP, muscle_img)
    
    fig, axs = plt.subplots(1,2, figsize=FIGSIZE)

    mframe.plot_kin_img(axs[0], cmap='gray')
    mframe.plot_muscle_img(axs[1], cmap='gray')

    for ax in axs:
        ax.axis('off')

    plt.show()


sleap_muscle_file = "labels_muscle_frames_femur_V1.000_muscle_frames_900_1399.analysis.h5"
path = sleap_dir / sleap_muscle_file

def get_muscle_locations_from_sleap_file(
        sleap_file_path,
        trochanter_id=0,
        femur_tibia_id=1,
        ):
    with h5py.File(sleap_file_path, 'r') as f:
        locations = f["tracks"][:].T

    muscle_tro = locations[:, trochanter_id, :, 0]
    muscle_ft = locations[:, femur_tibia_id, :, 0]
    return muscle_tro, muscle_ft

mtro_locations, mft_locations = get_muscle_locations_from_sleap_file(path)

# frame relative id: 173
idx = 174
def get_labeled_muscle_frame(
        npy_file_path,
        idx):
    muscle_frames = np.load(npy_file_path)

    labeled_muscle_frame = muscle_frames[idx]

    return labeled_muscle_frame

muscle_npy_file = 'muscle_frames_900_1399.npy'
muscle_frames_dir = pu.get_muscle_frames_dir()
muscle_file_path = muscle_frames_dir/ muscle_npy_file


mtro = mtro_locations[idx]
mft = mft_locations[idx]
m_axis = np.array([mtro, mft])
muscle_frame = get_labeled_muscle_frame(muscle_file_path, idx)

# get corresponding kin frame id
kin_to_muscle_div_factor = 4
rel_kin_idx = idx*kin_to_muscle_div_factor


kin_sleap_file = "label_femur_for_muscle_V1.000_900_1440_kin.analysis.h5"

kintro, kinft, _, topkintro, topkinft = get_locations_from_sleap_file(sleap_dir / kin_sleap_file)
kin_frame = get_labeled_kin_frame(kin_file_path, rel_kin_idx)
kinmidaxis = np.array([kintro[rel_kin_idx], kinft[rel_kin_idx]])
kintopaxis = np.array([topkintro[rel_kin_idx], topkinft[rel_kin_idx]])

def get_muscle_mframe():
    return mf.MappedFrame(
        kin_frame,
        kinmidaxis,
        kintopaxis,
        MMAP,
        muscle_frame,
        m_axis,
    )

def test_visualize_kin_muscle_with_axis():
    mframe = get_muscle_mframe()

    fig, axs = plt.subplots(1,2, figsize=FIGSIZE)

    mframe.plot_kin_img(axs[0], cmap='gray')
    mframe.plot_kin_middle_axis(axs[0])
    mframe.plot_kin_top_axis(axs[0])
    
    mframe.plot_muscle_img(axs[1], cmap='gray')
    mframe.plot_muscle_middle_axis(axs[1])

    print(mframe.get_muscle_middle_axis())

    for ax in axs:
        ax.axis('off')

    plt.show()

def test_visualize_muscle_oriented_scaled_map():
    mframe = get_muscle_mframe()

    mframe.align_map_axis_ref_point_on_muscle()
    mframe.orient_map_on_muscle()
    ratio = mframe.compute_muscle_map_ratio()
    print(ratio)
    mframe.scale_map(ratio=ratio)

    fig, ax = plt.subplots(1,2, figsize=FIGSIZE)

    delta = np.array([10, 0])

    mframe.plot_kin_img(ax[0], cmap='gray')
    mframe.plot_kin_middle_axis(ax[0])

    mframe.plot_muscle_img(ax[1], cmap='gray')
    mframe.plot_muscle_middle_axis(ax[1], delta=2*delta)
    mframe.plot_map_axis_middle_view_on_muscle(ax[1], delta=delta)
    mframe.plot_convex_hulls_on_middle_view(ax[1])
    

    for ax in ax:
        ax.axis('off')

    plt.show()






    

    
    





if __name__ == "__main__":
    
    ### Numerical tests ###
    test_compute_kin_middle_axis_angle()
    test_compute_kin_top_axis_angle()
    test_compute_kinematic_vector()

    
    ### Visualizations tests ###

    #test_plot_coordinate_frame()
    #test_visualize_map_aligned_with_kin_middle_axis()
    #test_visualize_scaled_translated_map()
    #test_visualize_compute_kinematic_vector()
    #test_visualize_orient_map_on_kin()
    #test_visualize_scaling_on_kinetic_frame()
    #test_visualize_muscle_img()

    #test_visualize_kin_muscle_with_axis()
    test_visualize_muscle_oriented_scaled_map()


    print("All tests passed.")