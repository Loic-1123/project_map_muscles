from _root_path import add_root
add_root()

import h5py
import tqdm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import map_muscles.path_utils as pu
import map_muscles.video_converter as vc
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.path_utils as pu

def print_sleap_labelling_info(sleap_file_path):
    with h5py.File(sleap_file_path, 'r') as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print("# frames, # nodes, # dimensions (x, y), # instances")
    print()

    print("===nodes: (id, name)===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

def write_kin_labeled_video(
        video_writer, figsize, 
        frames, frame_start, frame_end, 
        locations, labels,
        instance_id=0,
        n=1,
        s=5, colors=[np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])],
        ):

    print("===Creating video===")
    r = range(frame_start, frame_end, n)
    for i in tqdm.tqdm(r):
        frame = frames[i]

        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.imshow(frame, cmap='gray')

        for j, label in enumerate(labels):
            ax.scatter(
                locations[i, j, 0, instance_id], locations[i, j, 1, instance_id],
                color=colors[j], s=s, label=label
                )

        ax.set_title(f'frame {i}')
        ax.legend()

        # remove axis
        ax.axis('off')

        vc.save_frame_plt_film(video_writer, fig)

        # close figure
        plt.close(fig)

    vc.end_cv2_writing(out)
    print("Video created at ", out_path)
    print("===END: Creating video===")

if __name__ == "__main__":
    
    filename = 'label_femur_V1.000_900_1440_kin.analysis.h5'
    sleap_folder = pu.get_sleap_dir()
    file_path = sleap_folder / filename

    print_sleap_labelling_info(file_path)

    with h5py.File(file_path, 'r') as f:
        locations = f["tracks"][:].T

    mpl.rcParams['figure.figsize'] = [15,6]

    kin_path = pu.get_kin_dir()

    kin_frames = vc.extract_kin_frames(kin_path, 'jpg')

    kin_min_id = imu.get_min_id(kin_path, 'jpg')
    len_max = min(len(kin_frames), len(locations))

    figsize = (10, 5)
    out_path = pu.get_video_dir()
    video_name = 'check_labelling_on_kinematic_fps12.mp4'
    fps = 12
    out = vc.get_video_writer(video_name, figsize, fps=fps)

    frame_start = 710
    frame_end = len_max 

    labels = ['trochanter', 'femur-tibia', 'tibia-tarsus']
    write_kin_labeled_video(
        out, figsize, 
        kin_frames, frame_start, frame_end,
        locations, labels, n=1)
    
    
