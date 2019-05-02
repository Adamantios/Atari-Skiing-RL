from argparse import ArgumentParser, ArgumentTypeError
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

    # Reshape for batches and frames and return.
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
    filename_prefix = 'out/models/atari_skiing'
    plot_name_prefix = 'out/plots/atari_skiing'
    results_name_prefix = 'out/results/atari_skiing'
    save_interval = 100
    results_save_interval = 100
    info_interval_current = 20
    info_interval_mean = 100
    target_model_change = int(1e4)
    agent_path = ''
    agent_history = 4
    plot_train_results = True
    save_plot = True
    render = True
    downsample_scale = 2
    steps_per_action = 4
    episodes = int(1e4)
    epsilon = 1.
    final_epsilon = .1
    epsilon_decay = float(1e-4)
    total_observe_count = int(1e4)
    replay_memory_size = int(4e5)
    batch_size = 32
    gamma = .99
    optimizer_name = 'rmsprop'
    optimizer_choices = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
    learning_rate = float(25e-5)
    lr_decay = float(1e-6)
    beta1 = .9
    beta2 = .999
    rho = .95
    fuzz = .01
    momentum = .1

    def positive_int(value: any) -> int:
        """
        Checks if a value is a positive integer.

        :param value: the value to be checked.
        :return: the value if valid integer, otherwise raises an ArgumentTypeError.
        """
        int_value = int(value)

        if int_value <= 0:
            raise ArgumentTypeError("%s should be a positive integer value." % value)

        return int_value

    def positive_float(value: any) -> float:
        """
        Checks if a value is a positive float.

        :param value: the value to be checked.
        :return: the value if valid float, otherwise raises an ArgumentTypeError.
        """
        float_value = float(value)

        if float_value <= 0:
            raise ArgumentTypeError("%s should be a positive float value." % value)

        return float_value

    parser = ArgumentParser(description='Trains a DQN agent to play the Atari Skiing game.')

    parser.add_argument('-fp', '--filename_prefix', type=str, required=False, default=filename_prefix,
                        help='Filename prefix for the trained model to be saved (default %(default)s).')
    parser.add_argument('-hp', '--results_name_prefix', type=str, required=False, default=results_name_prefix,
                        help='Filename prefix for the loss history to be saved (default %(default)s).')
    parser.add_argument('-si', '--save_interval', type=positive_int, default=save_interval, required=False,
                        help='The save interval for the trained model (default %(default)s), in episodes.')
    parser.add_argument('-rsi', '--results_save_interval', type=int, default=results_save_interval, required=False,
                        help='The save interval for the results history (default %(default)s), in episodes.'
                             'Insert a negative value to not save the results history.')
    parser.add_argument('-iic', '--info_interval_current',
                        type=positive_int, default=info_interval_current, required=False,
                        help='The current scoring information interval (default %(default)s), in episodes.')
    parser.add_argument('-iim', '--info_interval_mean', type=positive_int, default=info_interval_mean, required=False,
                        help='The mean scoring information interval (default %(default)s), in episodes.')
    parser.add_argument('-ti', '--target_interval', type=positive_int, default=target_model_change, required=False,
                        help='The target model change interval (default %(default)s), in steps.')
    parser.add_argument('-a', '--agent', type=str, required=False, default=agent_path,
                        help='Filepath for a trained agent to be loaded (default %(default)s).')
    parser.add_argument('-ah', '--agent_history', type=positive_int, required=False, default=agent_history,
                        help='The agent\'s frame history (default %(default)s).')
    parser.add_argument('-np', '--no_plot', default=not plot_train_results, required=False, action='store_false',
                        help='Whether the train results should not be plot (default %(default)s).')
    parser.add_argument('-nsp', '--no_save_plot', default=not save_plot, required=False, action='store_false',
                        help='Whether the train results should not be saved (default %(default)s).')
    parser.add_argument('-p', '--plot_name', type=str, required=False, default=plot_name_prefix,
                        help='Filename prefix for the plots to be saved (default %(default)s).')
    parser.add_argument('-nr', '--no_render', default=not render, required=False, action='store_false',
                        help='Whether the environment should not be rendered (default %(default)s).')
    parser.add_argument('-d', '--downsample', type=positive_int, default=downsample_scale, required=False,
                        help='The downsampling scale to be used (default %(default)s).')
    parser.add_argument('-s', '--steps', type=positive_int, default=steps_per_action, required=False,
                        help='The steps to skip per action (default %(default)s).')
    parser.add_argument('-e', '--episodes', type=positive_int, default=episodes, required=False,
                        help='The episodes to run the training procedure (default %(default)s).')
    parser.add_argument('-eps', '--epsilon', type=positive_float, default=epsilon, required=False,
                        help='The epsilon for the e-greedy policy (default %(default)s).')
    parser.add_argument('-feps', '--final_epsilon', type=positive_float, default=final_epsilon, required=False,
                        help='The final epsilon for the e-greedy policy (default %(default)s).')
    parser.add_argument('-deps', '--decay', type=positive_float, default=epsilon_decay, required=False,
                        help='The epsilon decay for the e-greedy policy (default %(default)s).')
    parser.add_argument('-o', '--observe', type=positive_int, default=total_observe_count, required=False,
                        help='The total number of observing steps before the training begins, '
                             'thus taking random actions (default %(default)s).')
    parser.add_argument('-rm', '--replay_memory', type=positive_int, default=replay_memory_size, required=False,
                        help='The replay memory to be used for the agent (default %(default)s).')
    parser.add_argument('-b', '--batch', type=positive_int, default=batch_size, required=False,
                        help='The batch size to be randomly sampled from the memory for the training '
                             '(default %(default)s).')
    parser.add_argument('-g', '--gamma', type=positive_float, default=gamma, required=False,
                        help='The discount factor (default %(default)s).')
    parser.add_argument('-opt', '--optimizer', type=str.lower, default=optimizer_name, required=False,
                        choices=optimizer_choices,
                        help='The optimizer to be used. (default %(default)s).')
    parser.add_argument('-lr', '--learning_rate', type=positive_float, default=learning_rate, required=False,
                        help='The learning rate for the optimizer (default %(default)s).')
    parser.add_argument('-lrd', '--learning_rate_decay', type=positive_float, default=lr_decay, required=False,
                        help='The learning rate decay for the optimizer (default %(default)s).')
    parser.add_argument('-b1', '--beta1', type=positive_float, default=beta1, required=False,
                        help='The beta 1 for the optimizer (default %(default)s).')
    parser.add_argument('-b2', '--beta2', type=positive_float, default=beta2, required=False,
                        help='The beta 2 for the optimizer (default %(default)s).')
    parser.add_argument('-rho', type=positive_float, default=rho, required=False,
                        help='The rho for the optimizer (default %(default)s).')
    parser.add_argument('-f', '--fuzz', type=positive_float, default=fuzz, required=False,
                        help='The fuzz factor for the rmsprop optimizer (default %(default)s).')
    parser.add_argument('-m', '--momentum', type=positive_float, default=momentum, required=False,
                        help='The momentum for the optimizer (default %(default)s).')

    return parser
