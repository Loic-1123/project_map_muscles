from _root_path import add_root, get_root_path
add_root()

from pathlib import Path

root_path = Path(get_root_path())

map_muscles_dir = root_path / 'map_muscles'

data_dir = map_muscles_dir / 'data'

recording_dir = data_dir / '20240213_muscle_recording'

img_dir1 = recording_dir / '900_1440'
img_dir2 = recording_dir / '5760-6210'

video_dir = recording_dir / 'videos'

xray_dir = data_dir / 'xray'

directories = [
    data_dir, 
    recording_dir, 
    img_dir1, 
    img_dir2, 
    video_dir, 
    xray_dir]

for directory in directories:
    assert directory.exists(), f'Following directory does not exist: {directory}'

def get_map_muscles_dir():
    return map_muscles_dir

def get_data_dir():
    return data_dir

def get_recording_dir():
    return recording_dir

def get_img_dir(number='900_1440'):
    if number == '900_1440':
        return img_dir1
    elif number == '5760-6210':
        return img_dir2
    else:
        path = recording_dir / number
        assert path.exists(), f'Following path does not exist: {path}'
        return path

def get_kin_dir(number='900_1440'):
    path = get_img_dir(number) / 'kin'
    assert path.exists(), f'Following path does not exist: {path}'
    return path

def get_muscle_dir(number='900_1440'):
    path = get_img_dir(number) / 'muscle'
    assert path.exists(), f'Following path does not exist: {path}'
    return path
    

def get_video_dir():
    return video_dir

def get_xray_dir():
    return xray_dir