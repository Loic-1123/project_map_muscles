from _root_path import add_root
add_root()

import numpy as np
import pandas as pd

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

def generate_close_activities_vector(activities, r):
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


import numpy as np

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

    return np.where(losses < best_loss*ratio)[0]

def array_loss_function(
        mmap, 
        distance_loss_func=euclidean_distance_loss,
        loss_selector=remove_compare_to_best_idx,
        n=20, 
        max_iter=1000,
        plateau_max_iter=10,
        plateau_fraction=0.95,
        ):


    # fluorescence array of the image
    array = mmap.extract_fluorescence_array()
    # muscles pixels coordinates
    muscles_pixels_coordinates = mmap.extract_muscles_pixels_coordinates()


    # Prepare first iteration    
    ## generate activitiess0
    activitiess = generate_equally_spaced_activities(n, len(mmap.get_muscles()))
    r = 1/n

    plateau_counter = 0
    best_loss = np.inf

    # start iterations
    while max_iter > 0:
        max_iter -= 1

        # generate predictions
        predictions = [generate_linear_prediction(activities, muscles_pixels_coordinates, array.shape) for activities in activitiess]

        # calculate the loss for each prediction
        distance_losses = [distance_loss_func(array, prediction) for prediction in predictions]

        chosen_idx = loss_selector(distance_losses)

        # update
        ## update the activities
        selected_activitiess = activitiess[chosen_idx]
        new_activitiess = np.array([generate_close_activities_vector(activities, r) for activities in selected_activitiess])
        activitiess = np.unique(new_activitiess, axis=0)

        ## update the r
        r = r/2

        ## check for plateau: if best loss is not changing
        best_ratio = np.min(distance_losses)/best_loss
        if best_ratio > plateau_fraction:
            plateau_counter += 1
            if plateau_counter > plateau_max_iter:
                break
        else:
            plateau_counter = 0
            best_loss = np.min(distance_losses)

    # return sorted activitiess
    return activitiess[np.argsort(distance_losses)]


        

        

    

    
    

    

    

    



    


        






