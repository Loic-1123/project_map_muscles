from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import functools

root_path = Path(get_root_path())

map_muscles_dir = root_path / 'map_muscles'

data_dir = map_muscles_dir / 'data'

recording_dir = data_dir / '20240213_muscle_recording'

img_dir1 = recording_dir / '900_1440'
img_dir2 = recording_dir / '5760-6210'

kin_frames_dir = img_dir1/ 'kin_frames'

video_dir = recording_dir / 'videos'

xray_dir = data_dir / 'xray'

sleap_dir = map_muscles_dir / 'sleap'

map_dir = data_dir / 'muscles_maps'

basic_map_dir = map_dir / 'basic_map'

map_matching_dir = map_dir / 'map_matching'

"""
directories = [
    map_muscles_dir,
    data_dir, 
    recording_dir, 
    img_dir1, 
    img_dir2, 
    video_dir,
    xray_dir,
    sleap_dir,
    map_dir,
    basic_map_dir, 
    map_matching_dir]

for directory in directories:
    assert directory.exists(), f'Following directory does not exist: {directory}'
"""
def assert_directory(dir_path):
    assert dir_path.exists(), f'Following directory does not exist: {dir_path}'

def create_directory(dir_path, parents=True, exist_ok=True):
    dir_path.mkdir(parents=parents, exist_ok=exist_ok)
    print(f'Created directory: {dir_path}')

def assert_create_return_dir(dir_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(assert_dir=True, create_dir=True):
            if create_dir:
                create_directory(dir_path)
            if assert_dir:
             assert_directory(dir_path)
        
            return dir_path
        return wrapper
    return decorator

def get_map_muscles_dir():
    """
    Returns the directory path where the map muscles are stored.

    Returns:
        pathlib::Path : The directory path where the map muscles are stored.
    """
    return map_muscles_dir

def get_data_dir():
    return data_dir

def get_recording_dir():
    return recording_dir

def get_img_dir(number='900_1440'):
    """
    Returns the directory path for the specified image number.

    Parameters:
        number (str): The image number series. Default is '900_1440'.

    Returns:
        pathlib::Path : The directory path for the specified image number.

    Raises:
        AssertionError: If the specified path does not exist.
    """
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

def get_sleap_dir():
    return sleap_dir

def get_map_dir():
    return map_dir

def get_basic_map_dir():
    return basic_map_dir

@assert_create_return_dir(map_matching_dir)
def get_map_matching_dir():
    pass

@assert_create_return_dir(kin_frames_dir)
def get_kin_frames_dir():
    pass