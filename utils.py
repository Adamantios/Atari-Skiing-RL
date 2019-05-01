from argparse import ArgumentParser
from os import makedirs, path
import numpy as np


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Converts an rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def downsample(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Downsamples an image array, by a scale factor.

    :param img: the image to downsample.
    :param scale: the downsampling scale factor.
    :return: the downsampled image.
    """
    if scale < 2:
        return img

    return img[::scale, ::scale]


def atari_preprocess(frame_array: np.ndarray, downsample_scale: int = 2) -> np.ndarray:
    """
    Prepossesses the given atari frame array.

    :param frame_array: the atari frame array.
    :param downsample_scale: a scale to downsample the given array with.
    :return: the preprocessed frame array.
    """
    # Converting into greyscale since colors don't matter.
    greyscale_frame = rgb2gray(frame_array)

    # Downsampling the image.
    resized_frame = downsample(greyscale_frame, downsample_scale)

    # Reshape for batches and return.
    return resized_frame[np.newaxis, :, :, np.newaxis]


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = path.dirname(filepath)

    # Create directory if it does not exist
    if not path.exists(directory) and not directory == '':
        makedirs(directory)


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the atari skiing script.

    :return: ArgumentParser object.
    """
    filename_prefix = 'out/atari_skiing'
    save_interval = 100
    info_interval = 100
    target_model_change = 10000
    agent_path = ''
    plot_train_results = True
    render = True
    downsample_scale = 2
    steps_per_action = 4
    episodes = 10000
    epsilon = 1.
    final_epsilon = .1
    epsilon_decay = 1e-4
    total_observe_count = 1000
    replay_memory_size = 4e5
    batch_size = 32
    gamma = .99

    parser = ArgumentParser(description='Trains a DQN agent to play the Atari Skiing game.')

    parser.add_argument('-f', '--filename', type=str, required=False, default=filename_prefix,
                        help='Filename prefix for the trained model to be saved (default %(default)s).')
    parser.add_argument('-si', '--save_interval', type=int, default=save_interval, required=False,
                        help='The save interval for the trained model (default %(default)s), in episodes.')
    parser.add_argument('-ii', '--info_interval', type=int, default=info_interval, required=False,
                        help='The scoring information interval (default %(default)s), in episodes.')
    parser.add_argument('-ti', '--target_interval', type=int, default=target_model_change, required=False,
                        help='The target model change interval (default %(default)s), in steps.')
    parser.add_argument('-a', '--agent', type=str, required=False, default=agent_path,
                        help='Filepath for a trained agent to be loaded (default %(default)s).')
    parser.add_argument('-np', '--no_plot', default=not plot_train_results, required=False, action='store_false',
                        help='Whether the train results should not be plot (default %(default)s).')
    parser.add_argument('-nr', '--no_render', default=not render, required=False, action='store_false',
                        help='Whether the environment should not be rendered (default %(default)s).')
    parser.add_argument('-d', '--downsample', type=int, default=downsample_scale, required=False,
                        help='The downsampling scale to be used (default %(default)s).')
    parser.add_argument('-s', '--steps', type=int, default=steps_per_action, required=False,
                        help='The steps to skip per action (default %(default)s).')
    parser.add_argument('-e', '--episodes', type=int, default=episodes, required=False,
                        help='The episodes to run the training procedure (default %(default)s).')
    parser.add_argument('-eps', '--epsilon', type=int, default=epsilon, required=False,
                        help='The epsilon for the e-greedy policy (default %(default)s).')
    parser.add_argument('-feps', '--final_epsilon', type=int, default=final_epsilon, required=False,
                        help='The final epsilon for the e-greedy policy (default %(default)s).')
    parser.add_argument('-deps', '--decay', type=int, default=epsilon_decay, required=False,
                        help='The epsilon decay for the e-greedy policy (default %(default)s).')
    parser.add_argument('-o', '--observe', type=int, default=total_observe_count, required=False,
                        help='The total number of observing steps before the training begins, '
                             'thus taking random actions (default %(default)s).')
    parser.add_argument('-rm', '--replay_memory', type=int, default=replay_memory_size, required=False,
                        help='The replay memory to be used for the agent (default %(default)s).')
    parser.add_argument('-b', '--batch', type=int, default=batch_size, required=False,
                        help='The batch size to be randomly sampled from the memory for the training '
                             '(default %(default)s).')
    parser.add_argument('-g', '--gamma', type=int, default=gamma, required=False,
                        help='The discount factor (default %(default)s).')

    return parser
