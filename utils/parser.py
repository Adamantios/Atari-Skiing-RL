from argparse import ArgumentParser, ArgumentTypeError

FILENAME_PREFIX = 'out/models/atari_skiing'
PLOT_NAME_PREFIX = 'out/plots/atari_skiing'
RESULTS_NAME_PREFIX = 'out/results/atari_skiing'
SAVE_INTERVAL = 100
RESULTS_SAVE_INTERVAL = 100
INFO_INTERVAL_CURRENT = 20
INFO_INTERVAL_MEAN = 100
TARGET_MODEL_CHANGE = int(1E4)
AGENT_PATH = ''
AGENT_HISTORY = 4
PLOT_TRAIN_RESULTS = True
SAVE_PLOTS = True
RENDER = True
DOWNSAMPLE_SCALE = 2
STEPS_PER_ACTION = 4
FIT_FREQUENCY = 4
NO_OPERATION = 30
EPISODES = int(1E4)
EPSILON = 1.
FINAL_EPSILON = .1
EPSILON_DECAY = float(1E-4)
TOTAL_OBSERVE_COUNT = int(1E4)
REPLAY_MEMORY_SIZE = int(4E5)
BATCH_SIZE = 32
GAMMA = .99
OPTIMIZER_NAME = 'RMSProp'
OPTIMIZER_CHOICES = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
LEARNING_RATE = float(25E-5)
LR_DECAY = float(1E-6)
BETA1 = .9
BETA2 = .999
RHO = .95
FUZZ = .01
MOMENTUM = .1


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


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the atari skiing script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Trains a DQN agent to play the Atari Skiing game.')

    parser.add_argument('-fp', '--filename_prefix', type=str, required=False, default=FILENAME_PREFIX,
                        help='Filename prefix for the trained model to be saved (default %(default)s).')
    parser.add_argument('-rp', '--results_name_prefix', type=str, required=False, default=RESULTS_NAME_PREFIX,
                        help='Filename prefix for the results history to be saved (default %(default)s).')
    parser.add_argument('-si', '--save_interval', type=positive_int, default=SAVE_INTERVAL, required=False,
                        help='The save interval for the trained model (default %(default)s), in episodes.')
    parser.add_argument('-rsi', '--results_save_interval', type=int, default=RESULTS_SAVE_INTERVAL, required=False,
                        help='The save interval for the results history (default %(default)s), in episodes.'
                             'Insert a negative value to not save the results history.')
    parser.add_argument('-iic', '--info_interval_current',
                        type=positive_int, default=INFO_INTERVAL_CURRENT, required=False,
                        help='The current scoring information interval (default %(default)s), in episodes.')
    parser.add_argument('-iim', '--info_interval_mean', type=positive_int, default=INFO_INTERVAL_MEAN, required=False,
                        help='The mean scoring information interval (default %(default)s), in episodes.')
    parser.add_argument('-ti', '--target_interval', type=positive_int, default=TARGET_MODEL_CHANGE, required=False,
                        help='The target model change interval (default %(default)s), in steps.')
    parser.add_argument('-a', '--agent', type=str, required=False, default=AGENT_PATH,
                        help='Filepath for a trained agent to be loaded (default %(default)s).')
    parser.add_argument('-ah', '--agent_history', type=positive_int, required=False, default=AGENT_HISTORY,
                        help='The agent\'s frame history (default %(default)s).')
    parser.add_argument('-np', '--no_plot', default=not PLOT_TRAIN_RESULTS, required=False, action='store_true',
                        help='Whether the train results should not be plot.')
    parser.add_argument('-nsp', '--no_save_plots', default=not SAVE_PLOTS, required=False, action='store_true',
                        help='Whether the train results plots should not be saved.')
    parser.add_argument('-p', '--plot_name', type=str, required=False, default=PLOT_NAME_PREFIX,
                        help='Filename prefix for the plots to be saved (default %(default)s).')
    parser.add_argument('-nr', '--no_render', default=not RENDER, required=False, action='store_true',
                        help='Whether the environment should not be rendered.')
    parser.add_argument('-d', '--downsample', type=positive_int, default=DOWNSAMPLE_SCALE, required=False,
                        help='The downsampling scale to be used (default %(default)s).')
    parser.add_argument('-fs', '--frame_skipping', type=positive_int, default=STEPS_PER_ACTION, required=False,
                        help='The frames to skip per action (default %(default)s).')
    parser.add_argument('-ff', '--fit_frequency', type=positive_int, default=FIT_FREQUENCY, required=False,
                        help='The actions to take between an agent\'s fit (default %(default)s).')
    parser.add_argument('-no', '--no_operation', type=positive_int, default=NO_OPERATION, required=False,
                        help='The maximum number of no operation steps at the beginning of the game '
                             '(default %(default)s).')
    parser.add_argument('-e', '--episodes', type=positive_int, default=EPISODES, required=False,
                        help='The episodes to run the training procedure (default %(default)s).')
    parser.add_argument('-eps', '--epsilon', type=positive_float, default=EPSILON, required=False,
                        help='The epsilon for the e-greedy policy (default %(default)s).')
    parser.add_argument('-feps', '--final_epsilon', type=positive_float, default=FINAL_EPSILON, required=False,
                        help='The final epsilon for the e-greedy policy (default %(default)s).')
    parser.add_argument('-deps', '--decay', type=positive_float, default=EPSILON_DECAY, required=False,
                        help='The epsilon decay for the e-greedy policy (default %(default)s).')
    parser.add_argument('-o', '--observe', type=positive_int, default=TOTAL_OBSERVE_COUNT, required=False,
                        help='The total number of observing steps before the training begins, '
                             'thus taking random actions (default %(default)s).')
    parser.add_argument('-rm', '--replay_memory', type=positive_int, default=REPLAY_MEMORY_SIZE, required=False,
                        help='The replay memory to be used for the agent (default %(default)s).')
    parser.add_argument('-b', '--batch', type=positive_int, default=BATCH_SIZE, required=False,
                        help='The batch size to be randomly sampled from the memory for the training '
                             '(default %(default)s).')
    parser.add_argument('-g', '--gamma', type=positive_float, default=GAMMA, required=False,
                        help='The discount factor (default %(default)s).')
    parser.add_argument('-opt', '--optimizer', type=str.lower, default=OPTIMIZER_NAME, required=False,
                        choices=OPTIMIZER_CHOICES,
                        help='The optimizer to be used. (default %(default)s).')
    parser.add_argument('-lr', '--learning_rate', type=positive_float, default=LEARNING_RATE, required=False,
                        help='The learning rate for the optimizer (default %(default)s).')
    parser.add_argument('-lrd', '--learning_rate_decay', type=positive_float, default=LR_DECAY, required=False,
                        help='The learning rate decay for the optimizer (default %(default)s).')
    parser.add_argument('-b1', '--beta1', type=positive_float, default=BETA1, required=False,
                        help='The beta 1 for the optimizer (default %(default)s).')
    parser.add_argument('-b2', '--beta2', type=positive_float, default=BETA2, required=False,
                        help='The beta 2 for the optimizer (default %(default)s).')
    parser.add_argument('-rho', type=positive_float, default=RHO, required=False,
                        help='The rho for the optimizer (default %(default)s).')
    parser.add_argument('-f', '--fuzz', type=positive_float, default=FUZZ, required=False,
                        help='The fuzz factor for the rmsprop optimizer (default %(default)s).')
    parser.add_argument('-m', '--momentum', type=positive_float, default=MOMENTUM, required=False,
                        help='The momentum for the optimizer (default %(default)s).')

    return parser
