from os import path
from typing import Union
from warnings import warn

from collections import deque

from keras.optimizers import adam, rmsprop, sgd, adagrad, adadelta, adamax

from core.agent import EGreedyPolicy, DQN
from core.game import Game
from core.model import atari_skiing_model, huber_loss, frame_can_pass_the_net, min_frame_dim_that_passes_net

from utils.os_operations import create_path
from utils.parser import create_parser
from utils.plotting import Plotter

from utils.scoring import Scorer


def run_checks() -> None:
    """ Checks the input arguments. """
    # Set default variables.
    poor_observe = bad_target_model_change = 500
    frame_history_ceiling = 10

    # Create the path to the files, if necessary.
    create_path(agent_name_prefix)
    create_path(plots_name_prefix)
    create_path(results_name_prefix)

    if info_interval_mean == 1:
        warn('Info interval mean has no point to be 1. '
             'The program will continue, but the means will be ignored.'.format(info_interval_mean))

    if target_model_change < bad_target_model_change:
        warn('Target model change is extremely small ({}). This will possibly make the agent unstable.'
             'Consider a value greater than {}'.format(target_model_change, bad_target_model_change))

    if not path.exists(agent_path) and agent_path != '':
        raise FileNotFoundError('File {} not found.'.format(agent_path))

    if agent_frame_history > frame_history_ceiling:
        warn('The agent\'s frame history is too big ({}). This will possibly make the agent unstable and slower.'
             'Consider a value smaller than {}'.format(agent_frame_history, frame_history_ceiling))

    if downsample_scale == 1:
        warn('Downsample scale set to 1. This means that the atari frames will not be scaled down.')

    # Downsampling should result with at least 32 pixels on each dimension,
    # because the first convolutional layer has a filter 8x8 with stride 4x4.
    if not frame_can_pass_the_net(game.observation_space_shape[0], game.observation_space_shape[1]):
        raise ValueError('Downsample is too big. It can be set from 1 to {}'
                         .format(min(int(game.pixel_rows / min_frame_dim_that_passes_net()),
                                     int(game.pixel_columns / min_frame_dim_that_passes_net()))))

    if plot_train_results and episodes == 1:
        warn('Cannot plot for 1 episode only.')

    if epsilon > 1:
        raise ValueError('Epsilon cannot be set to a greater value than 1.'
                         'Got {}'.format(epsilon))

    if final_epsilon > 1:
        raise ValueError('Epsilon cannot be set to a greater value than 1.'
                         'Got {}'.format(final_epsilon))

    if final_epsilon > epsilon:
        raise ValueError('Final epsilon ({}) cannot be greater than epsilon ({}).'
                         .format(final_epsilon, epsilon))

    if epsilon_decay > epsilon - final_epsilon:
        warn('Epsilon decay is too big ({})!'.format(epsilon_decay))

    if total_observe_count < poor_observe:
        warn('The total number of observing steps ({}) is too small and could bring poor results.'
             'Consider a value grater than {}'.format(total_observe_count, poor_observe))


def initialize_optimizer() -> Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]:
    """
    Initializes an optimizer based on the user's choices.

    :return: the optimizer.
    """
    if optimizer_name == 'adam':
        return adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=lr_decay)
    elif optimizer_name == 'rmsprop':
        return rmsprop(lr=learning_rate, rho=rho, epsilon=fuzz)
    elif optimizer_name == 'sgd':
        return sgd(lr=learning_rate, momentum=momentum, decay=lr_decay)
    elif optimizer_name == 'adagrad':
        return adagrad(lr=learning_rate, decay=lr_decay)
    elif optimizer_name == 'adadelta':
        return adadelta(lr=learning_rate, rho=rho, decay=lr_decay)
    elif optimizer_name == 'adamax':
        return adamax(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=lr_decay)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')


def create_agent() -> DQN:
    """
    Creates the atari skiing agent.

    :return: the agent.
    """
    if agent_path != '':
        # Load the model and the memory.
        model, memory = DQN.load_agent(agent_path, {'huber_loss': huber_loss})
        print('Agent {} has been loaded successfully.'.format(agent_path))
    else:
        # Init the model.
        model = atari_skiing_model(game.observation_space_shape, game.action_space_size, optimizer)
        # Create the replay memory for the agent.
        memory = deque(maxlen=replay_memory_size)

    # Create the policy.
    policy = EGreedyPolicy(epsilon, final_epsilon, epsilon_decay, total_observe_count, game.action_space_size)
    # Create the agent.
    dqn = DQN(model, target_model_change, memory, gamma, batch_size, game.observation_space_shape,
              game.action_space_size, policy)

    return dqn


def save_agent(episode: int) -> None:
    """
    Saves the agent after a certain episode.

    :param episode: the episode.
    """
    if episode % save_interval == 0 or save_interval == 1:
        print('Saving agent.')
        filename = agent.save_agent("{}_{}".format(agent_name_prefix, episode))
        print('Agent has been successfully saved as {}.'.format(filename))


def end_of_episode_actions(episode: int) -> None:
    """
    Take actions after the episode finishes.

    Show scoring information and saves the model.

    :param episode: the episode for which the actions will be taken.
    """
    save_agent(episode)

    if episode % info_interval_current == 0 or info_interval_current == 1:
        scorer.show_episode_scoring(episode)

    if info_interval_mean > 1 and episode % info_interval_mean == 0:
        scorer.show_mean_scoring(episode)

    if episode == episodes and episodes > 1:
        plotter.plot_scores_vs_episodes(scorer.max_scores, scorer.total_scores)
        plotter.plot_loss_vs_episodes(scorer.huber_loss_history)

    if results_save_interval > 0 and (episodes % results_save_interval == 0 or results_save_interval == 1):
        scorer.save_results(episode)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    agent_name_prefix = args.filename_prefix
    results_name_prefix = args.results_name_prefix
    results_save_interval = args.results_save_interval
    save_interval = args.save_interval
    info_interval_current = args.info_interval_current
    info_interval_mean = args.info_interval_mean
    target_model_change = args.target_interval
    agent_path = args.agent
    agent_frame_history = args.agent_history
    plot_train_results = not args.no_plot
    save_plot = not args.no_save_plot
    plots_name_prefix = args.plot_name
    render = not args.no_render
    downsample_scale = args.downsample
    # TODO steps_per_action = args.steps
    # TODO add fit frequency.
    # TODO add maximum no operation at start argument.
    episodes = args.episodes
    epsilon = args.epsilon
    final_epsilon = args.final_epsilon
    epsilon_decay = args.decay
    total_observe_count = args.observe
    replay_memory_size = args.replay_memory
    batch_size = args.batch
    gamma = args.gamma
    optimizer_name = args.optimizer
    learning_rate = args.learning_rate
    lr_decay = args.learning_rate_decay
    beta1 = args.beta1
    beta2 = args.beta2
    rho = args.rho
    fuzz = args.fuzz
    momentum = args.momentum

    # Create a scorer.
    scorer = Scorer(episodes, info_interval_mean, results_name_prefix)

    # Create the game.
    game = Game(episodes, render, downsample_scale, scorer, agent_frame_history)

    # Check arguments.
    run_checks()

    # Create the optimizer.
    optimizer = initialize_optimizer()

    # Create the agent.
    agent = create_agent()

    # Create a plotter.
    plotter = Plotter(episodes, plots_name_prefix, plot_train_results, save_plot)

    # Start the game loop.
    for finished_episode in game.play_game(agent):
        # Take specific actions after the end of each episode.
        end_of_episode_actions(finished_episode)
