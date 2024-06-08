from _root_path import add_root, get_root_path
add_root()

from pathlib import Path
import functools

"""
This file contains utility functions to handle paths of the project (mainly directories).
They assertain the existence of the directories and creates them if they do not exist.
"""

root_path = Path(get_root_path())

map_muscles_dir = root_path / 'map_muscles'

data_dir = map_muscles_dir / 'data'

recording_dir = data_dir / '20240213_muscle_recording'

img_dir1 = recording_dir / '900_1440'
img_dir2 = recording_dir / '5760-6210'

kin_frames_dir = img_dir1/ 'kin_frames'

muscle_frames_dir = img_dir1 / 'muscle_frames'

video_dir = recording_dir / 'videos'

xray_dir = data_dir / 'xray'

sleap_dir = map_muscles_dir / 'sleap'

map_dir = data_dir / 'muscles_maps'

basic_map_dir = map_dir / 'basic_map'

lf_map_dir = map_dir / 'lf_leg_map'

lh_map_dir = map_dir / 'lh_leg_map'

rm_map_dir = map_dir / 'rm_leg_map'

map_matching_dir = map_dir / 'map_matching'

def assert_directory(dir_path):
    """
    Asserts that the given directory path exists.

    Parameters:
    dir_path (Path): The path to the directory.

    Raises:
    AssertionError: If the directory does not exist.

    """
    assert dir_path.exists(), f'Following directory does not exist: {dir_path}'

def create_directory(dir_path, parents=True, exist_ok=True):
    """
    Create a directory at the specified path if it doesn't already exist.

    Args:
        dir_path (str or Path): The path of the directory to be created.
        parents (bool, optional): If True, create parent directories as needed. Defaults to True.
        exist_ok (bool, optional): If True, do not raise an exception if the directory already exists. Defaults to True.
    """
    if not dir_path.exists():
        dir_path.mkdir(parents=parents, exist_ok=exist_ok)
        print(f'Created directory: {dir_path}')
    
def assert_create_return_dir(dir_path):
    """
    A decorator that asserts the existence of a directory and creates it if necessary.
    
    Args:
        dir_path (str): The path of the directory to be created.
    
    Returns:
        function: The decorated function.
    """
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

@assert_create_return_dir(map_muscles_dir)
def get_map_muscles_dir():
    """
    Returns the directory path of the map_muscles project.
    """
    pass
@assert_create_return_dir(data_dir)
def get_data_dir():
    pass

@assert_create_return_dir(recording_dir)
def get_recording_dir():
    pass

def get_img_dir(number='900_1440', create=False, assert_dir=True):
    """
    Returns the directory path for the specified image number.

    Parameters:
        number (str): The image number series. Default is '900_1440'.

    Returns:
        pathlib::Path : The directory path for the specified image number.

    Raises:
        AssertionError: If the specified path does not exist.
    """
    if number == '900_1440': path = img_dir1
    elif number == '5760-6210': path = img_dir2
    else: path = recording_dir / number

    if create: create_directory(path)
    if assert_dir: assert_directory(path)    

    return path

def get_kin_dir(number='900_1440', create=False, assert_dir=True):
    path = get_img_dir(number) / 'kin'
    if create: create_directory(path)
    if assert_dir: assert_directory(path)
    return path

def get_muscle_dir(number='900_1440', create=False, assert_dir=True):
    path = get_img_dir(number) / 'muscle'
    if create: create_directory(path)
    if assert_dir: assert_directory(path)
    return path

@assert_create_return_dir(video_dir)    
def get_video_dir():
    pass

@assert_create_return_dir(xray_dir)
def get_xray_dir():
    pass

@assert_create_return_dir(sleap_dir)
def get_sleap_dir():
    pass

@assert_create_return_dir(map_dir)
def get_map_dir():
    pass

@assert_create_return_dir(basic_map_dir)
def get_basic_map_dir():
    pass

@assert_create_return_dir(map_matching_dir)
def get_map_matching_dir():
    pass

@assert_create_return_dir(kin_frames_dir)
def get_kin_frames_dir():
    pass

@assert_create_return_dir(muscle_frames_dir)
def get_muscle_frames_dir():
    pass


@assert_create_return_dir(lf_map_dir)
def get_lf_map_dir():
    pass

@assert_create_return_dir(lh_map_dir)
def get_lh_map_dir():
    pass

@assert_create_return_dir(rm_map_dir)
def get_rm_map_dir():
    pass