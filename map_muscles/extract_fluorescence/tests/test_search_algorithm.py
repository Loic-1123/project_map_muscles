from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt

import map_muscles.extract_fluorescence.mapping.euler_mapped_frame as mf
import map_muscles.extract_fluorescence.mapping.search_algorithm as sa
from map_muscles.extract_fluorescence.tests.test_euler_mapped_frame import get_muscle_mframe

def test_visualize_generate_linear_prediction():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    activities = np.zeros(len(mframe.mmap.get_muscles()))

    activated_muscles_idx = [0, 2, 4]

    for idx in activated_muscles_idx:
        activities[idx] = 1

    coordinates = mframe.extract_muscles_pixels_coordinates()

    muscle_img_shape = mframe.get_muscle_img().shape

    prediction = sa.generate_linear_prediction(activities, coordinates, muscle_img_shape)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    mframe.plot_muscle_img(axs[0])
    mframe.plot_convex_hulls_on_middle_view(axs[0])

    # plot the linear prediction
    
    mframe.plot_convex_hulls_on_middle_view(axs[1], activated_muscles_idx)
    axs[1].imshow(prediction)

    for ax in axs:
        ax.axis('off')

    plt.show()

def test_visualize_generate_equally_spaced_activities():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    n_muscles = len(mframe.mmap.get_muscles())
    activitiess = sa.generate_equally_spaced_activities(4, n_muscles)

    coordinates = mframe.extract_muscles_pixels_coordinates()

    muscle_img_shape = mframe.get_muscle_img().shape

    predictions = [sa.generate_linear_prediction(activities, coordinates, muscle_img_shape) for activities in activitiess]

    show_n = 5

    # set seed
    np.random.seed(0)
    idx_to_show = np.random.choice(len(predictions), show_n, replace=False)

    fig, axs = plt.subplots(1, show_n, figsize=(5*show_n, 5))

    for idx, ax in zip(idx_to_show, axs):
        ax.imshow(predictions[idx])
        ax.axis('off')
        mframe.plot_convex_hulls_on_middle_view(ax)
        
        # indicate the muscle activities
        activities = activitiess[idx]

        # set small title indicating muscle activities
        title = f"Activities: {activities}"
        ax.set_title(title)

        # small font
        ax.title.set_fontsize(6)

        fig.colorbar(ax.imshow(predictions[idx]), ax=ax, shrink=0.5)

    plt.show()

def test_visualize_generate_close_activities_vector():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    activities = np.random.rand(len(mframe.mmap.get_muscles()))

    new_activitiess = sa.generate_close_activities_vector(activities, 0.5)

    coordinates = mframe.extract_muscles_pixels_coordinates()

    muscle_img_shape = mframe.get_muscle_img().shape

    predictions = [sa.generate_linear_prediction(activities, coordinates, muscle_img_shape) for activities in new_activitiess]

    show_n = 5

    # set seed
    np.random.seed(0)

    muscle_img_shape = mframe.get_muscle_img().shape

    idx_to_show = np.random.choice(len(predictions), show_n, replace=False)

    fig, axs = plt.subplots(1, show_n+1, figsize=(5*show_n, 5))

    img_activities = sa.generate_linear_prediction(activities, coordinates, muscle_img_shape)
    axs[0].imshow(img_activities)
    axs[0].axis('off')
    mframe.plot_convex_hulls_on_middle_view(axs[0])
    axs[0].set_title(f'Original activities: {activities.round(2)}')
    axs[0].title.set_fontsize(6)
    fig.colorbar(axs[0].imshow(img_activities), ax=axs[0], shrink=0.5)

    axs = axs[1:]

    for idx, ax in zip(idx_to_show, axs):
        ax.imshow(predictions[idx])
        ax.axis('off')
        mframe.plot_convex_hulls_on_middle_view(ax)
        
        # indicate the muscle activities
        activities = new_activitiess[idx]
        # set small title indicating muscle activities
        title = f"Activities: {activities.round(2)}"
        ax.set_title(title)

        # small font
        ax.title.set_fontsize(6)

        fig.colorbar(ax.imshow(predictions[idx]), ax=ax, shrink=0.5)

    plt.show()

def test_visualize_extract_fluorescence_array():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    array = mframe.extract_fluorescence_array()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.imshow(array)

    mframe.plot_convex_hulls_on_middle_view(ax)

    ax.axis('off')

    plt.show()

def test_remove_worse():
    def assert_remove_worse(losses, idx, fraction):
        # assert that the idx has the correct length
        assert len(idx) == int(fraction*len(losses)), \
            f"Expected idx to have length {int(fraction*len(losses))}, got {len(idx)}"

        # assert that all results are smaller than 1*fraction
        assert np.all(losses[idx] < 1*fraction), \
            f"Expected all losses to be smaller than 1*fraction, got {np.sort(losses[idx])}"

    losses1 = np.linspace(0, 1, 100)

    losses2 = np.random.rand(100)

    lossess = [losses1, losses2]

    for losses in lossess:
        for fraction in [0.1, 0.5, 0.7]:
            idx = sa.remove_worse_idx(losses, fraction)
            assert_remove_worse(losses, idx, fraction)

def test_remove_compare_to_best():
    def assert_remove_compare_to_best(losses, idx, ratio):
        # assert that the idx has the correct length
        assert len(idx) == np.sum(losses < np.min(losses)*ratio), \
            f"Expected idx to have length {np.sum(losses < np.min(losses)*ratio)}, got {len(idx)}"

        # assert that all results are smaller than 1*fraction
        assert np.all(losses[idx] < np.min(losses)*ratio), \
            f"Expected all losses to be smaller than 1*fraction, got {np.sort(losses[idx])}"

    losses1 = np.linspace(0, 1, 100)

    losses2 = np.random.rand(100)

    lossess = [losses1, losses2]
    ratios = [1.1, 1.5, 2.0, 20, 50]

    for losses in lossess:
        for ratio in ratios:
            idx = sa.remove_compare_to_best_idx(losses, ratio)
            assert_remove_compare_to_best(losses, idx, ratio)

if __name__ == '__main__':
    
    test_remove_worse()
    test_remove_compare_to_best()
    
    
    #test_visualize_generate_linear_prediction()
    #test_visualize_generate_equally_spaced_activities()
    #test_visualize_generate_close_activities_vector()
    #test_visualize_extract_fluorescence_array()

    print("All tests passed")










    
    
