from _root_path import add_root
add_root()

import numpy as np
import pandas as pd
import tqdm

import map_muscles.extract_fluorescence.mapping.euler_mapped_frame as mf
from map_muscles.extract_fluorescence.tests.test_euler_mapped_frame import get_muscle_mframe



# greedy algorithm to find opimized last orientation parameter (gamm/roll)

GAMMA_RANGE = 2*np.pi

def generate_equally_spaced_gammas(n):
    """
    Generate equally spaced gamma values.

    Parameters:
        n (int): The number of gamma values to generate.

    Returns:
        numpy.ndarray: An array of equally spaced gamma values ranging from 0 to 2*pi.
    """

    return np.linspace(0, GAMMA_RANGE, n)

def generate_gammas(gamma, r, keep_gamma=True):
    """
    Generate gamma values around a given gamma value.

    Parameters:
        gamma (float): The gamma value around which to generate the gamma values.
        r (float): The range of gamma values to generate.
        keep_gamma (bool): Flag indicating whether to keep the original gamma value (default: True).

    Returns:
        numpy.ndarray: An array of gamma values around the given gamma value.
    """

    gamma1 = (gamma - r) % GAMMA_RANGE
    gamma2 = (gamma + r) % GAMMA_RANGE

    if keep_gamma:
        return np.array([gamma1, gamma, gamma2])
    else:
        return np.array([gamma1, gamma2])


def optimize_gamma(
        mframe: mf.MappedFrame, 
        loss_function,
        n=24, n_iter=1000, keep_best=5,
        ):
    
    # prepare the first iteration
    gammas = generate_equally_spaced_gammas(n)
    losses = []
    r = (GAMMA_RANGE / (n-1)) / 2

    for _ in range(n_iter):
        for gamma in gammas:
            mframe.roll_map_to_gamma(gamma)
            loss = loss_function(mframe)

            losses.append(loss)
        
        best_gammas = best_gammas[np.argsort(losses)[:keep_best]]

        new_gammas = []
        for gamma in best_gammas:
            new_gammas.extend(generate_gammas(gamma, r))

        # prepare new iteration
        gammas = new_gammas
        losses.clear()
        r = r/2


def generate_linear_prediction(
        muscles_activities, 
        muscles_pixels_coordinates,
        img_shape,
        gain=1
        ):
    
    # for each muscle, create a matrix of shape img_shape
    # where the value of the given pixel coordinates is the muscle activity
    muscles_predictions = []
    for muscle_activity, coordinates in zip(muscles_activities, muscles_pixels_coordinates):
        muscle_prediction = np.zeros(img_shape)
        # (n, 2), n is the number of pixels, 2 is the coordinates
        muscle_prediction[coordinates[:, 1], coordinates[:, 0]] = muscle_activity
        muscles_predictions.append(muscle_prediction)
    
    muscles_predictions = np.array(muscles_predictions)*gain
    
    # sum all the muscle predictions
    return np.sum(muscles_predictions, axis=0)

def generate_equally_spaced_activities(n_values, n_muscles):
    """
    Generate equally spaced activities for a given number of values and muscles.

    Parameters:
    - n_values (int): The number of different equally spaced values to generate.
    - n_muscles (int): The number of muscles.

    Returns:
    - combinations (ndarray): An array of shape (n_values^n_muscles, n_muscles) containing the equally spaced activities.
    """

    values = np.linspace(0, 1, n_values)
    
    grid = np.meshgrid(*[values]*n_muscles)
    
    combinations = np.array(grid).reshape(n_muscles, -1).T

    return combinations

def generate_all_close_activities_vector(activities, r):
    """
    Generate a new vector of activities by adding combinations of -r, 0, and r to the input vector.

    Parameters:
    activities (numpy.ndarray): The input vector of activities.
    r (float): The value to be added to the input vector.

    Returns:
    numpy.ndarray: The new vector of activities obtained by adding combinations of -r, 0, and r to the input vector.
    shape = (3**n, n), n is the number of muscles (aka the number of activities)
    """

    # prepare the combinations to be added: -r, 0, r
    n = len(activities)

    grid = np.meshgrid(*([[-r, 0, r]]*n))
    grid = np.array(grid).reshape(n, -1).T

    new_activities = grid + activities

    # clamp the values to be between 0 and 1
    new_activities = np.clip(new_activities, 0, 1)

    return new_activities

def generate_one_step_close_activities_vector(activities, r):
    """
    Generate a new vector of activities by adding combinations of -r, 0, and r to the input vector.

    Parameters:
    activities (numpy.ndarray): The input vector of activities.
    r (float): The value to be added to the input vector.

    Returns:
    numpy.ndarray: The new vector of activities obtained by adding combinations of -r, 0, r to the input vector.
    shape = (3*n, n), n is the number of muscles (aka the number of activities)
    """

    # prepare the combinations to be added: -r, 0, r
    n = len(activities)

    r_minus = np.diag([-r]*n)
    r_plus = np.diag([r]*n)

    new_activities = np.vstack([activities, activities + r_minus, activities + r_plus])

    # clamp the values to be between 0 and 1
    new_activities = np.clip(new_activities, 0, 1)

    return new_activities


def euclidean_distance_loss(array1, array2):
    """
    Calculate the Euclidean distance loss between two arrays.

    Parameters:
    - array1 (numpy.ndarray): The first array.
    - array2 (numpy.ndarray): The second array.

    Returns:
    - float: The Euclidean distance loss between the two arrays.
    """

    return np.linalg.norm(array1 - array2)

def remove_worse_idx(losses, fraction=0.5):
    """
    Remove the worst fraction of losses and return the indices of the remaining losses.

    Parameters:
    - losses (numpy.ndarray): Array of loss values.
    - fraction (float): Fraction of losses to remove. Default is 0.5.

    Returns:
    - numpy.ndarray: Array of indices corresponding to the remaining losses.
    """

    sorted_idx = np.argsort(losses)
    return sorted_idx[:int(fraction * len(losses))]
   
def remove_compare_to_best_idx(losses, ratio=1.5):
    """
    return the idx of the losses that are better than the best loss by a ratio
    """

    best_loss = np.min(losses)

    return np.where(losses <= best_loss*ratio)[0]

def array_loss_function(
        img_array,
        muscles_pixels_coordinates,
        distance_loss_func=euclidean_distance_loss,
        loss_selector=remove_compare_to_best_idx,
        nb_activities_threshold=1000,
        activities_generator=generate_one_step_close_activities_vector,
        n=3, 
        max_iter=50,
        plateau_max_iter=10,
        plateau_fraction=1.05,
        yield_iter=False,
        threshold_warning=False
        ):
    """
    Calculates the loss function an array of pixels values,
    compared to the possible configuration of muscle activities.

    This function returns the best prediction of the muscle activities that would map 
    on the input image array.

    Parameters:
    - img_array: The input image array of pixel values to compare with.
    - muscles_pixels_coordinates: The coordinates of the muscles pixels, that the area taken by each muscle on the image.
    - distance_loss_func: The distance loss function to be used. Default is euclidean_distance_loss.
    - loss_selector: The loss selector function to be used to choose the best idx. Default is remove_compare_to_best_idx.
    - nb_activities_threshold: The threshold for the number of activities. Default is 1000.
    - activities_generator: The activities generator function. Default is generate_one_step_close_activities_vector.
    - n: The number of equally spaced values for 1 dimension used to generate the firt round of activities. Default is 3, aka [0., 0.5, 1.].
    - max_iter: The maximum number of iterations. Default is 500.
    - plateau_max_iter: The maximum number of iterations to wait for a plateau. Default is 10.
    - plateau_fraction: The fraction of the best loss to consider as a plateau. Default is 1.05.
    - yield_iter: Whether to yield intermediate results at each iteration. Default is False.
    - threshold_warning: Whether to print a warning when the number of activities exceeds the threshold. Default is False.

    Returns:
    - activitiess: The generated activities, sorted from best to worse of the last iteration.
    - distance_losses: The calculated distance losses, sorted too.
    - predictions: The corresponding generated predictions, sorted too. Could be used to plot the best predictions as an img.
    - r: The value of r.
    - iter_count: The id of the iteration.
    """

    # Prepare first iteration
    n_muscles = len(muscles_pixels_coordinates)
    activitiess = generate_equally_spaced_activities(n, n_muscles)
    predictions = []
    distance_losses = []
    r = 1/n

    plateau_counter = 0
    best_loss = np.inf

    iter_count = 1
    
    def log_decaying_frac2(i):
        return 1 + 1/np.log(i+2)**2


    # start iterations
    for _ in tqdm.tqdm(range(max_iter)):

        # generate predictions
        predictions = np.array([generate_linear_prediction(activities, muscles_pixels_coordinates, img_array.shape) for activities in activitiess])

        # calculate the loss for each prediction
        distance_losses = np.array([distance_loss_func(img_array, prediction) for prediction in predictions])
        chosen_idx = loss_selector(distance_losses, ratio=log_decaying_frac2(iter_count))

        #preventing exponential growth of the number of activities
        if len(chosen_idx) > nb_activities_threshold:
            # select best activities
            if threshold_warning:
                str_reached = "Nb activities threshold reached."
                str_nb_activies = f"Chosen nb activities for next round = {len(chosen_idx)}"
                str_clamped = f"Clamping, keeping the best {nb_activities_threshold} activities."
                warning_str = f"{str_reached}\n{str_nb_activies}\n{str_clamped}"
                print(warning_str)

            chosen_idx = np.argsort(distance_losses)[:nb_activities_threshold]

        if yield_iter:
            yield activitiess, distance_losses, predictions, r, iter_count

        # if last iteration, break without updating
        if iter_count == max_iter:
            break

        # update

        ## check for plateau: if best loss is not changing
        best_ratio = best_loss/np.min(distance_losses)
        if best_ratio > plateau_fraction:
            plateau_counter += 1
            if plateau_counter > plateau_max_iter or np.isclose(best_loss, 0):
                break
        else:
            plateau_counter = 0

        ## update r: update every n_muscles iterations, to enable reaching [+r, +r, +r, ...]
        # since the one step activities generation change one dimension at a time
        if (iter_count % n_muscles) == 0:
            r = r/2
        
        ## update best loss
        best_loss = np.min(distance_losses)

        ## update the activities
        selected_activitiess = activitiess[chosen_idx]
        new_activitiess = np.array([activities_generator(activities, r) for activities in selected_activitiess])
        new_activitiess = new_activitiess.reshape(-1, len(muscles_pixels_coordinates))

        ### rounding to enable the removal of duplicates (due to float numbers)
        rounded_activitiess = np.round(new_activitiess, decimals=4)
        activitiess = np.unique(rounded_activitiess, axis=0)

        ## update itreation counter
        iter_count += 1

    # return sorted 
    sorted_idx = np.argsort(distance_losses)
    activitiess = activitiess[sorted_idx]
    distance_losses = distance_losses[sorted_idx]
    predictions = np.array(predictions)[sorted_idx]
    
    return activitiess, distance_losses, predictions, r, iter_count

def generate_random_activities(n, n_muscles, seed=0):
    """
    Generate random activities for muscles.

    Parameters:
    - n (int): Number of activities to generate.
    - n_muscles (int): Number of muscles.
    - seed (int): Seed for random number generation. Default is 0.
    - noise (bool): Whether to add noise to the generated activities. Default is True.

    Returns:
    - activitiess (ndarray): Array of shape (n, n_muscles) containing the generated activities.
    - noised_activitiess (ndarray): Array of shape (n, n_muscles) containing the generated activities with added noise.
    """

    np.random.seed(seed)

    activitiess = np.random.rand(n, n_muscles)
    
    return activitiess

def least_squares_activities(img_array, imgs_bool):
    """
    Calculate the least squares activities for a given image array and boolean images.

    Parameters:
    img_array (ndarray): The input image array.
    imgs_bool (list): The list of boolean images representing the pixels where a muscle is located on the image (True = 1).

    Returns:
    ndarray: The least squares activities.

    """
    A = np.array([img.flatten() for img in imgs_bool]).T
    B = img_array.flatten().reshape(-1, 1)
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    return x.reshape(-1)

def lstsq_activities_loss(
    img_array,
    imgs_bool,
    loss_function=euclidean_distance_loss,
    ):

    activities = least_squares_activities(img_array, imgs_bool)
    predicted_img = activities @ np.array([img.flatten() for img in imgs_bool])
    predicted_img = predicted_img.flatten()
    img = img_array.flatten()

    return loss_function(img, predicted_img), activities




