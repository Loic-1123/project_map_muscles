from _root_path import add_root, get_root_path
add_root()

import video_converter as vc
from pathlib import Path

"""
Test the video converter functions from video_converter.py.
"""


root_path = Path(get_root_path())
image_folder = root_path / 'map_muscles' / 'data' / '20240213_muscle_recording'
output_folder = image_folder / 'videos'

kin_folder = 'kin'
muscle_folder = 'muscle'

number = '900_1440'

img_kin_folder = image_folder / number / kin_folder
img_muscle_folder = image_folder / number / muscle_folder
    
assert img_kin_folder.exists(), f"{img_kin_folder} does not exist."
assert img_muscle_folder.exists(), f"{img_muscle_folder} does not exist."

video_folder =  image_folder / 'videos'
video_folder.mkdir(exist_ok=True)

def test_kin_index_filtering(
        img_folder=img_kin_folder, 
        output_folder=output_folder,
        name = 'kin_test_index_bounds',
        fps=30):
    # create video with index bounds
    vc.write_kin_video(img_folder, name, output_folder, fps=fps, start_index=4000, end_index=5000)

def test_muscle_index_filtering(
        img_folder=img_kin_folder, 
        output_folder=output_folder,
        name = 'test_muscle_index_bounds',
        fps=30):
    # create video with index bounds
    vc.write_muscle_video(img_folder, name, output_folder, fps=fps, start_index=1100, end_index=1200)

def test_write_kin_video(img_folder=img_kin_folder, output_folder=output_folder):
    # create 1 fps video
    name = 'test_kin_1fps'
    vc.write_kin_video(img_folder, name, output_folder, fps=1)
    # create 30 fps video
    name = 'test_kin_30fps'
    vc.write_kin_video(img_folder, name, output_folder, fps=30)
    # create video with index bounds
    test_kin_index_filtering(img_folder, output_folder)

def test_write_muscle_video(img_folder=img_muscle_folder, output_folder=output_folder):
    # create 30 fps video
    name = 'test_muscle_30fps'
    vc.write_muscle_video(img_folder, name, output_folder, fps=30, gain=1)
    # create video with index bounds
    test_muscle_index_filtering(img_folder, output_folder, fps=30)

if __name__ == "__main__":

    test_write_kin_video()

    test_write_muscle_video()

    print("END of test_video_converter.py")
