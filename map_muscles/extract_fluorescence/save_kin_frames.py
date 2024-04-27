from _root_path import add_root
add_root()

import numpy as np

import map_muscles.path_utils as pu
import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.video_converter as vc

def save_kin_frames(
        save_dir: str = pu.get_kin_frames_dir(),
        kin_img_dir: str = pu.get_kin_dir()
    ):

    frames = vc.extract_kin_frames(kin_img_dir, 'jpg')

    # get min id of the frames
    min_id = imu.get_min_id(pu.get_kin_dir(), 'jpg')
    #get max id of the frames
    max_id = min_id + len(frames)

    name = f'kin_frames_{min_id}_{max_id}.npy'

    # save the frames as npy files
    np.save(save_dir / name, frames)

def load_kin_frames(
        file_path: str
    ):

    return np.load(file_path)


if __name__ == "__main__":

    save_dir = pu.get_kin_frames_dir()

    kin_img_dir = pu.get_kin_dir()

    frames = vc.extract_kin_frames(kin_img_dir, 'jpg')

    # get min id of the frames
    min_id = imu.get_min_id(pu.get_kin_dir(), 'jpg')
    #get max id of the frames
    max_id = min_id + len(frames)

    name = f'kin_frames_{min_id}_{max_id}.npy'

    # save the frames as npy files
    np.save(save_dir / name, frames)




        



