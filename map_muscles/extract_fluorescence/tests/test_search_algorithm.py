from _root_path import add_root
add_root()

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import map_muscles.extract_fluorescence.mapping.euler_mapped_frame as mf
import map_muscles.extract_fluorescence.mapping.search_algorithm as sa
from map_muscles.extract_fluorescence.tests.test_euler_mapped_frame import get_muscle_mframe
import map_muscles.path_utils as pu

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

def test_visualize_generate_all_close_activities_vector():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    activities = np.random.rand(len(mframe.mmap.get_muscles()))

    new_activitiess = sa.generate_all_close_activities_vector(activities, 0.5)

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

def test_euclidian_distance_loss():

    mframe = get_muscle_mframe()
    mframe.prepare_map()

    shape = mframe.get_muscle_img().shape

    v = np.random.randn(*shape)

    assert sa.euclidean_distance_loss(v, v) == 0, "Expected the loss(v, v) to be 0"

    v2 = np.random.randn(*shape)

    assert sa.euclidean_distance_loss(v, v2) > 0, "Expected the loss(v, v2) to be > 0"

    v3 = np.zeros(shape)

    assert np.isclose(sa.euclidean_distance_loss(v, v3), np.linalg.norm(v)) , "Expected the euclidian loss(v, 0) to be the norm of v3"

    v4 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    v5 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    assert np.isclose(sa.euclidean_distance_loss(v4, v5), np.sqrt(2)), f"Expected the euclidian loss(v4, v5) to be 2, got {sa.euclidean_distance_loss(v4, v5)}"

def demo_array_loss_function(
        max_iter =100,
        m = 10,
        modulo = 10
):
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    np.random.seed(0)
    truth_activities = np.random.rand(len(mframe.mmap.get_muscles()))

    coordinates = mframe.extract_muscles_pixels_coordinates()
    shape = mframe.get_muscle_img().shape

    truth_array = sa.generate_linear_prediction(truth_activities, coordinates, shape)

    print("=== DEMO array_loss_function ===")

    print("MAX ITER: ", max_iter)



    values = sa.array_loss_function(
        truth_array, coordinates, 
        max_iter=max_iter, 
        yield_iter=True, threshold_warning=True,
    )

    for activitiess, distance_losses, predictions, r, iter_count in values:
        if (iter_count % modulo == 0) or (iter_count == max_iter):
            idx = np.argsort(distance_losses)
            print(f"---Iteration {iter_count}---\n")
            print(f"Target activities: {truth_activities}\n")
            print(f"Best losses: \n{distance_losses[idx[:m]]}\n")
            print(f"Best activities: \n{activitiess[idx[:m]]}, activitiess.shape: { activitiess.shape}\n")
            print(f"r: {r}")
            print("------")

def stats_on_array_loss_function(
    n=30,
    max_iter=50,
    fig_name = "mean_losses_per_target_with_noise.png",
):

    mframe = get_muscle_mframe()
    mframe.prepare_map()

    n_muscles = len(mframe.mmap.get_muscles())
    coordinates = mframe.extract_muscles_pixels_coordinates()
    shape = mframe.get_muscle_img().shape

    target_activitiess = sa.generate_random_activities(n, n_muscles)

    tqdm1 = tqdm.tqdm(
        target_activitiess, 
        leave=True, 
        total=n,
        desc="Iterating over targets"
        )

    target_mean_losses_per_target = []
    noised_mean_losses_per_target = []

    for target_activities in tqdm1:
        target_prediction = sa.generate_linear_prediction(target_activities, coordinates, shape)
        noise = np.random.randn(*shape)
        noised_prediction = target_prediction + noise

        values = sa.array_loss_function(
            target_prediction, coordinates, 
            max_iter=max_iter, 
            yield_iter=True, threshold_warning=False,
        )

        mean_best_losses = []

        tqdm2 = tqdm.tqdm(values, leave=False, desc="Iterating over losses: target")

        for activitiess, distance_losses, predictions, r, iter_count in tqdm2:
            idx = np.argsort(distance_losses)
            mean_best_losses.append(np.mean(distance_losses))

        mean_best_losses = np.array(mean_best_losses)
        target_mean_losses_per_target.append(mean_best_losses)

        noised_values = sa.array_loss_function(
            noised_prediction, coordinates, 
            max_iter=max_iter, 
            yield_iter=True, threshold_warning=False,
        )

        noised_mean_losses = []

        for activitiess, distance_losses, predictions, r, iter_count in noised_values:
            losses = np.array([sa.euclidean_distance_loss(noised_prediction, prediction) for prediction in predictions])
            noised_mean_losses.append(np.mean(losses))

        noised_mean_losses = np.array(noised_mean_losses)
        noised_mean_losses_per_target.append(noised_mean_losses)


    target_mean_losses_per_target = np.array(target_mean_losses_per_target)
    noised_mean_losses_per_target = np.array(noised_mean_losses_per_target)

    # plot the mean losses per target, as scatter plot

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for mean_losses, noised_losses in zip(target_mean_losses_per_target, noised_mean_losses_per_target):
        axs[0].scatter(np.arange(len(mean_losses)), mean_losses, alpha=0.1, color=[0,0,0])
        axs[1].scatter(np.arange(len(noised_losses)), noised_losses, alpha=0.1, color=[1,0,0])

    axs[0].set_title("Mean losses per target")
    axs[1].set_title("Mean losses per noised target, compared to unnoised target")
    
    for ax in axs:        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean loss")

    save_dir = pu.get_data_dir() / "search_algorithm"
    save_dir.mkdir(exist_ok=True)

    plt.savefig(save_dir / fig_name)
    plt.show()

def test_visualize_MappedFrame_extract_imgs_bool():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    imgs_bool = mframe.extract_imgs_bool()

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    ax.imshow(imgs_bool[0]+imgs_bool[1])

    mframe.plot_convex_hulls_on_middle_view(ax)

    ax.axis('off')

    plt.show()

def test_least_squares_activities(n=30, seed=0):
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    n_muscles = len(mframe.mmap.get_muscles())
    coordinates = mframe.extract_muscles_pixels_coordinates()
    shape = mframe.get_muscle_img().shape
    imgs_bool = mframe.extract_imgs_bool()

    np.random.seed(seed)

    tqdmn = tqdm.tqdm(
        range(n), 
        leave=True, 
        total=n,
        desc=f"test_least_squares(): testing least squares {n} times"
        )

    for _ in tqdmn:

        target_activities = np.random.rand(n_muscles)

        target_prediction = sa.generate_linear_prediction(target_activities, coordinates, shape)

        lsqu_activities = sa.least_squares_activities(target_prediction, imgs_bool)

        # round to 4 decimals
        target_activities = np.round(target_activities, 4)
        lsqu_activities = np.round(lsqu_activities, 4)

        assert np.allclose(target_activities, lsqu_activities), \
            f"Expected the target activities to be the same as the least squares activities, got {target_activities} and {lsqu_activities}"

def test_lstsq_activities_loss(n=30):
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    n_muscles = len(mframe.mmap.get_muscles())
    coordinates = mframe.extract_muscles_pixels_coordinates()
    shape = mframe.get_muscle_img().shape
    imgs_bool = mframe.extract_imgs_bool()

    tqdmn = tqdm.tqdm(
        range(n),
        leave=True,
        total=n,
        desc=f"test_lstsq_loss(): testing least squares loss {n} times"
    )

    for _ in tqdmn:
        target_activities = np.random.rand(n_muscles)

        target_prediction = sa.generate_linear_prediction(target_activities, coordinates, shape)

        loss, _ = sa.lstsq_activities_loss(target_prediction, imgs_bool)

        assert np.isclose(loss, 0),\
            f"Expected the loss to be 0, got {loss}"

def demo_optimize_gamma():
    mframe = get_muscle_mframe()
    mframe.prepare_map()

    np.random.seed(0)
    target_gamma = np.random.rand()*2*np.pi
    target_activities = np.random.rand(len(mframe.mmap.get_muscles()))

    mframe.roll_map_to_gamma(target_gamma)

    flat_imgs_bool = mframe.extract_flatten_imgs_bool()

    target_img_array = (target_activities @ flat_imgs_bool).reshape(mframe.get_muscle_img().shape)

    gammas, losses, activitiess, r, iter_count = sa.optimize_gamma(
        mframe, target_img_array, n_gammas = 60)


    # plot 5 best predictions
    n=5
    fig, axs = plt.subplots(1, n+1, figsize=(5*(n+1), 5))

    # plot the target prediction
    mframe.roll_map_to_gamma(target_gamma)
    axs[0].imshow(target_img_array)
    mframe.plot_convex_hulls_on_middle_view(axs[0])
    axs[0].set_title(f"Target prediction: gamma: {target_gamma:.2f}, activities: {target_activities.round(2)}")
    axs[0].title.set_fontsize(6)
    axs[0].axis('off')

    for i in range(0,n):
        ax = axs[i+1]
        gamma = gammas[i]
        mframe.roll_map_to_gamma(gamma)
        prediction = (activitiess[i] @ mframe.extract_flatten_imgs_bool()).reshape(mframe.get_muscle_img().shape)
        ax.imshow(prediction)
        mframe.plot_convex_hulls_on_middle_view(ax)

        title = f"Gamma: {gamma:.2f}, Loss: {losses[i]:.2f}, Activities: {activitiess[i].round(2)}"
        ax.set_title(title)
        # reduce title size
        ax.title.set_fontsize(6)
        
        ax.axis('off')

    plt.show()




if __name__ == '__main__':
    
    #test_remove_worse()
    #test_remove_compare_to_best()
    #test_euclidian_distance_loss()
    #test_least_squares_activities()
    #test_lstsq_activities_loss()

    #demo_array_loss_function()
    #stats_on_array_loss_function()
    demo_optimize_gamma()
    
    
    
    #test_visualize_generate_linear_prediction()
    #test_visualize_generate_equally_spaced_activities()
    #test_visualize_generate_all_close_activities_vector()
    #test_visualize_extract_fluorescence_array()


    print("All tests passed")