from _root_path import add_root
add_root()

import cv2
import os
import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import map_muscles.extract_fluorescence.imaging_utils as imu
import map_muscles.path_utils as pu

"""
This file contains functions to:
- extract frames/images from folders,
- write video from frames
- write video from matplotlib figures
"""

def get_fourcc(ext = 'mp4v'):

    return cv2.VideoWriter_fourcc(*ext)

def get_video_dimensions(figsize, factor=100):
    """Returns the video dimensions based on the figsize and factor.

    Args:
        figsize (tuple): The size of the figure (width, height).
        factor (int, optional): The scaling factor. Defaults to 100.

    Returns:
        tuple: The video dimensions (width, height).
    """
    return (int(figsize[0]*factor), int(figsize[1]*factor))

def get_video_writer(
        video_name,
        figsize,
        fps,
        video_dir = pu.get_video_dir(),
        fourcc=get_fourcc(), 
        ):
    """Returns a cv2 VideoWriter object.

    Args:
        video_name (str): The name of the output video file.
        figsize (tuple): The size of the figure (width, height).
        video_dir (Path object, optional): The directory where the video will be saved. Defaults to get_video_dir().
        fourcc (str, optional): The four character code of the video codec. Defaults to get_fourcc().
        fps (int, optional): The frames per second of the video. Defaults to 6.

    Returns:
        cv2.VideoWriter: The video writer object.
    """

    output_file = str(video_dir/ video_name)
    video_dimensions = get_video_dimensions(figsize)
    return cv2.VideoWriter(output_file, fourcc, fps, video_dimensions)

def frames_to_video(
        frames, 
        video_file, 
        fps:int=1,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        ):

    frame = frames[0]
    height, width = frame.shape
    
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height), isColor=False) # for gray scale

    print(f'Creating video with {len(frames)} frames to {video_file}')

    tqdm_frames = tqdm.tqdm(frames)
    for frame in tqdm_frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    assert video_file.exists(), f'video {video_file} was not created'
    print(f'Video created to {video_file}')

get_index = lambda path: int(path.split('.')[0])

def extract_img_names(img_folder, img_extension):
    """
    Returns:
        list[str]: list of image names
    """
    return [img for img in os.listdir(img_folder) if img.endswith(img_extension)]

def extract_img_paths(img_folder, img_extension):
    """
    Extracts the paths of all image files in the given folder with the specified extension.

    Args:
        img_folder (str): The path to the folder containing the image files.
        img_extension (str): The extension of the image files to be extracted.

    Returns:
        list: A list of paths to the image files.

    """
    return [img_folder / img for img in os.listdir(img_folder) if img.endswith(img_extension)]

def print_index_filtering(start_index, end_index):
    """
    Prints the index filtering applied to images.

    Parameters:
        start_index (int or None): The start index for filtering.
        end_index (int or None): The end index for filtering.

    Returns:
        None
    """

    if start_index is None and end_index is None:
        print('No index filtering applied')
    elif start_index is None:
        print(f'Filtering images with end_index: {end_index}')
    elif end_index is None:
        print(f'Filtering images with start_index: {start_index}')
    else:
        print(f'Filtering images with start_index: {start_index} and end_index: {end_index}')

def index_filtering(images_paths, start_index, end_index):
    """
    Filters a list of image paths based on the start and end index.

    Args:
        images_paths (list): A list of image paths.
        start_index (int): The start index for filtering.
        end_index (int): The end index for filtering.

    Returns:
        list: A filtered list of image paths.

    """
    if start_index:
        images_paths = [path for path in images_paths if (get_index(path) > start_index)]
    if end_index:
        images_paths = [path for path in images_paths if (get_index(path) < end_index)]
    
    print_index_filtering(start_index, end_index)
    return images_paths

def extract_kin_frames(kin_path, img_extension='jpg', start_index=None, end_index=None):
    """
    Extracts frames from a folder containing kinematic images.
    Applies a filtering based on the start and end index, if provided.

    Args:
        kin_path (str): The path to the directory.
        img_extension (str, optional): The image extension of the frames. Defaults to 'jpg'.
        start_index (int, optional): The starting index of the frames to extract. Defaults to None.
        end_index (int, optional): The ending index of the frames to extract. Defaults to None.

    Returns:
        list: A list of extracted frames.
    """
    images_paths = extract_img_names(kin_path, img_extension)
    images_paths = index_filtering(images_paths, start_index, end_index)
    images_paths.sort(key=get_index)
    images = [cv2.imread(str(kin_path / image_path)) for image_path in images_paths]
    frames = []

    nb_frames = len(images)*3

    print(f'Extracting {nb_frames} frames from {kin_path}')
    tqdm_images = tqdm.tqdm(images)
    for image in tqdm_images:
        frames.append(image[:,:,0])
        frames.append(image[:,:,1])
        frames.append(image[:,:,2])

    return frames

def extract_muscle_frames(muscle_path, img_extension='tif', start_index=None, end_index=None, gain=1):
    """
    Extracts muscle frames from the given directory containing muscle images.

    Args:
        muscle_path (str): The path to the directory.
        img_extension (str, optional): The extension of the muscle images. Defaults to 'tif'.
        start_index (int, optional): The starting index of the muscle frames to extract. Defaults to None.
        end_index (int, optional): The ending index of the muscle frames to extract. Defaults to None.
        gain (int, optional): The gain to apply to the extracted muscle frames. Defaults to 1.

    Returns:
        list: A list of extracted muscle frames.
    """
    images_paths = extract_img_names(muscle_path, img_extension)
    images_paths = index_filtering(images_paths, start_index, end_index)
    images_paths.sort(key=lambda x: int(x.split('.')[0]))
    images = [cv2.imread(str(muscle_path / image_path), -1)*gain for image_path in images_paths]
    return images

def write_kin_video(
        img_folder,
        video_name,
        output_folder,
        start_index=None,
        end_index=None,
        img_extension='jpg',
        fps=1
        ):
    """
    Writes a kinematic video from a folder of images.

    Args:
        img_folder (str or Path): Path to the folder containing the kinetic images.
        video_name (str): Name of the output video file (without extension).
        output_folder (str or Path): Path to the folder where the output video will be saved.
        start_index (int, optional): Start index for filtering the images. Defaults to None.
        end_index (int, optional): End index for filtering the images. Defaults to None.
        img_extension (str, optional): Extension of the input image files. Defaults to 'jpg'.
        fps (int, optional): Frames per second for the output video. Defaults to 1.

    Raises:
        AssertionError: If the input image folder or output folder does not exist or is not a folder.

    """
    assert img_folder.exists() & img_folder.is_dir(), \
        f'img folder {img_folder} does not exist or is not a folder'

    assert output_folder.exists() & output_folder.is_dir(), \
        f'output folder {output_folder} does not exist or is not a folder'

    video = video_name + '.mp4'
    video_file = output_folder / video

    images_paths = [img for img in os.listdir(img_folder) if img.endswith(img_extension)]

    # filter out with index
    images_paths = index_filtering(images_paths, start_index, end_index)
    # make sure that the images are sorted
    images_paths.sort(key=get_index)

    print(f'Extracting frames from {img_folder} to create video')

    frames = []

    tqdm_images_paths = tqdm.tqdm(images_paths)
    for image in tqdm_images_paths:
        image_path = img_folder / image
        image = cv2.imread(str(image_path))
        
        frames.append(image[:,:,0])
        frames.append(image[:,:,1])
        frames.append(image[:,:,2])

    frames_to_video(frames, video_file, fps=fps)
    
def write_muscle_video(
        img_folder,
        video_name,
        output_folder,
        start_index = None,
        end_index = None,
        img_extension = 'tif',
        fps=1,
        gain=1,
        ):
    assert img_folder.exists() & img_folder.is_dir(), \
        f'img folder {img_folder} does not exist or is not a folder'
    
    assert output_folder.exists() & output_folder.is_dir(), \
        f'output folder {output_folder} does not exist or is not a folder'
    
    video = video_name + '.mp4'
    video_file = output_folder / video

    images_paths = extract_img_names(img_folder, img_extension)

    #filter out with index
    images_paths = index_filtering(images_paths, start_index, end_index)

    # make sure that the images are sorted
    images_paths.sort(key=lambda x: int(x.split('.')[0]))

    print(f'Extracting frames from {img_folder} to create video')

    frames = []

    tqdm_images_paths = tqdm.tqdm(images_paths)
    for image in tqdm_images_paths:
        image_path = img_folder / image
        image = cv2.imread(str(image_path), -1)*gain

        frames.append(image)

    print("shape: " + str(frames[0].shape))

    frames_to_video(frames, video_file, fps)

def write_the_two_complete_kin_videos(
        image_folder,
        output_folder,
        kin_folder= 'kin',
        number = '900_1440',
        number2 = '5760-6210',
        fps=1,
    ):

    """
    Writes a muscle video from a folder of images.

    Args:
        img_folder (Path): The path to the folder containing the images.
        video_name (str): The name of the output video file.
        output_folder (Path): The path to the folder where the video will be saved.
        start_index (int, optional): The starting index of the images to include in the video. Defaults to None.
        end_index (int, optional): The ending index of the images to include in the video. Defaults to None.
        img_extension (str, optional): The extension of the image files. Defaults to 'tif'.
        fps (int, optional): The frames per second of the output video. Defaults to 1.
        gain (int, optional): The gain to apply to the images. Defaults to 1.

    Raises:
        AssertionError: If the img_folder or output_folder does not exist or is not a folder.

    Returns:
        None
    """
    
    img_folder = image_folder / number / kin_folder
    img_folder2 = image_folder / number2 / kin_folder
    
    assert img_folder.exists()
    assert img_folder2.exists()

    video_folder =  image_folder / 'videos'
    video_folder.mkdir(exist_ok=True)

    video_name = number + '_' + kin_folder
    video_name2 = number2 + '_' + kin_folder

    write_kin_video(img_folder, video_name, output_folder, fps=fps)
    write_kin_video(img_folder2, video_name2, output_folder, fps=fps)

def save_frame_plt_film(out, fig):
    """
    Save (append) a frame from a matplotlib figure to a video file.

    Parameters:
    - out: cv2.VideoWriter
        The video writer object to write the frame to.
    - fig: matplotlib.figure.Figure
        The matplotlib figure object containing the frame to be saved.
    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    out.write(mat)

def end_cv2_writing(out):
    """Ends the cv2 video writing process.

    Args:
        out (cv2.VideoWriter): The video writer object.
    """
    out.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":

    output_folder = pu.get_video_dir()
    img_kin_folder = pu.get_kin_dir()
    img_muscle_folder = pu.get_muscle_dir()
    
    assert img_kin_folder.exists()
    assert img_muscle_folder.exists()

    # 24 - 29 sec on 30 fps video

    fps = 30

    start_sec = 24
    end_sec = 29

    kin_min_index = imu.get_min_id(img_kin_folder, 'jpg')

    kin_start_id = kin_min_index + start_sec * fps

    kin_end_id = kin_min_index + end_sec * fps

    kin_fps = 20
    
    write_kin_video(
        img_kin_folder, 'kinematic_clip', output_folder,
        start_index=kin_start_id, end_index=kin_end_id, fps=kin_fps
        )

    muscle_min_id = imu.get_min_id(img_muscle_folder, 'tif', id_format='{:06d}')

    ratio = 4

    start_muscle_index = imu.get_matching_muscle_id(kin_start_id, kin_min_index, ratio, muscle_min_id)
    end_muscle_index = imu.get_matching_muscle_id(kin_end_id, kin_min_index, ratio, muscle_min_id)

    muscle_fps = kin_fps//ratio

    write_muscle_video(img_muscle_folder, 'muscle_clip', output_folder, start_index=start_muscle_index, end_index=end_muscle_index, fps=muscle_fps, gain=1)

    kin_diff = kin_end_id - kin_start_id
    muscle_diff = end_muscle_index - start_muscle_index