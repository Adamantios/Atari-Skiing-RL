from os import path
import gym
from collections import deque
import numpy as np
from keras.optimizers import RMSprop
from math import inf
from agent import EGreedyPolicy, DQN
from model import atari_skiing_model, huber_loss
from utils import create_path, atari_preprocess, create_parser
import matplotlib.pyplot as plt


def run_checks() -> None:
    """ Checks the default variables. """
    # TODO add more checks.
    # Create the path to the file, if necessary.
    create_path(filename_prefix)

    if not path.exists(agent_path) and agent_path != '':
        raise FileNotFoundError('File {} not found.'.format(agent_path))

    if batch_size > total_observe_count:
        raise ValueError('Batch size ({}) should be less or equal with the total_observe_count ({}).'
                         .format(batch_size, total_observe_count))


def create_skiing_environment():
    """
    Creates a skiing environment.

    :return: the skiing environment, the initial state, the image's height and width and the action space's size.
    """
    # Create the skiing environment.
    environment = gym.make('Skiing-v0')
    # Reset the environment and get the initial state.
    init_state = environment.reset()
    # Get the observation space's height and width.
    height, width = environment.observation_space.shape[0], environment.observation_space.shape[1]
    # Get the number of possible moves.
    act_space_size = environment.action_space.n

    return environment, init_state, height, width, act_space_size


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
        model = atari_skiing_model(observation_space_shape, action_space_size, optimizer)
        # Create the replay memory for the agent.
        memory = deque(maxlen=replay_memory_size)

    # Create the policy.
    policy = EGreedyPolicy(epsilon, final_epsilon, epsilon_decay, total_observe_count, action_space_size)
    # Create the agent.
    dqn = DQN(model, target_model_change, memory, gamma, batch_size, observation_space_shape, action_space_size, policy)

    return dqn


def render_frame() -> None:
    """ Renders a frame, only if the user has chosen to do so. """
    if render:
        env.render()


def save_agent(episode: int) -> None:
    """
    Saves the agent after a certain episode.

    :param episode: the episode.
    """
    if episode % save_interval == 0 or save_interval < 2:
        print('Saving agent.')
        filename = agent.save_agent("{}_{}".format(filename_prefix, episode))
        print('Agent has been successfully saved as {}.'.format(filename))


def show_episode_scoring(episode: int) -> None:
    """
    Shows an episode's scoring information.

    :param episode: the episode for which the actions will be taken.
    """
    if episode % info_interval_current == 0 or info_interval_current == 1:
        # Print the episode's scores.
        print("Max score for the episode {} is: {} ".format(episode, max_scores[episode - 1]))
        print("Total score for the episode {} is: {} ".format(episode, total_scores[episode - 1]))


def show_mean_scoring(episode: int) -> None:
    """
    Shows the mean scoring information for the current episode.

    :param episode: the episode.
    """
    if info_interval_mean > 1 and episode % info_interval_mean == 0:
        # Print the episodes mean scores.
        mean_max_score = max_scores[episode - info_interval_mean:episode].sum() / info_interval_mean
        mean_total_score = total_scores[episode - info_interval_mean:episode].sum() / info_interval_mean

        print("Mean Max score for {}-{} episodes is: {} "
              .format(episode - info_interval_mean, episode, mean_max_score))
        print("Mean Total score for {}-{} episodes is: {} "
              .format(episode - info_interval_mean, episode, mean_total_score))


def plot_scores_vs_episodes(episode: int) -> None:
    """
    Plots scores vs episodes.

    :param episode: the current episode.
    """
    if plot_train_results and episode == episodes:
        fig = plt.figure(figsize=(12, 10))
        # Start from 1, not 0.
        plt.xlim(1, episodes)
        plt.plot(np.append(np.roll(max_scores, 1), max_scores[episodes - 1]))
        plt.plot(np.append(np.roll(total_scores, 1), total_scores[episodes - 1]))
        plt.xticks(range(1, episodes + 1))
        plt.title('Scores vs Episodes', fontsize='x-large')
        plt.xlabel('Episode', fontsize='large')
        plt.ylabel('Score', fontsize='large')
        plt.legend(['Max Score', 'Total Score'], loc='upper left', fontsize='large')
        fig.savefig(plot_name)

        if save_plot:
            plt.show()

        # TODO plot mean huber error per episode vs episodes.


def end_of_episode_actions(episode: int) -> None:
    """
    Take actions after the episode finishes.

    Show scoring information and saves the model.

    :param episode: the episode for which the actions will be taken.
    """
    save_agent(episode)
    show_episode_scoring(episode)
    show_mean_scoring(episode)
    plot_scores_vs_episodes(episode)


def game_loop() -> None:
    """ Starts the game loop and trains the agent. """
    # Run for a number of episodes.
    for episode in range(1, episodes + 1):
        # Init vars.
        max_score, total_score, done = -inf, 0, False

        # Reset and render the environment.
        current_state = env.reset()
        render_frame()

        # Preprocess current_state.
        current_state = atari_preprocess(current_state, downsample_scale)

        # Create preceding frames, using the starting frame.
        current_state = np.stack(tuple([current_state for _ in range(agent_frame_history)]), axis=2)

        # Set current state with the stacked.
        current_state = np.reshape(current_state,
                                   (1,
                                    pixel_rows // downsample_scale,
                                    pixel_columns // downsample_scale,
                                    agent_frame_history))

        while not done:
            # Take an action, using the policy.
            action = agent.take_action(current_state)
            # Take a step, using the action.
            next_state, reward, done, _ = env.step(action)
            # Render the frame.
            render_frame()

            # Preprocess the state.
            next_state = atari_preprocess(next_state, downsample_scale)
            # Append the frame history.
            next_state = np.append(next_state, current_state[:, :, :, :agent_frame_history - 1], axis=3)

            # Save sample <s,a,r,s'> to the replay memory.
            agent.append_to_memory(current_state, action, reward, next_state)
            # Fit agent.
            agent.fit()

            # Add reward to the total score.
            total_score += reward
            # Set current state with the next.
            current_state = next_state
            # Set max score.
            max_score = max(max_score, reward)

        # Add scores to the scores arrays.
        max_scores[episode - 1] = max_score
        total_scores[episode - 1] = total_score

        # Take end of episode specific actions.
        end_of_episode_actions(episode)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    filename_prefix = args.filename
    save_interval = args.save_interval
    info_interval_current = args.info_interval_current
    info_interval_mean = args.info_interval_mean
    target_model_change = args.target_interval
    agent_path = args.agent
    agent_frame_history = args.agent_history
    plot_train_results = not args.no_plot
    save_plot = not args.no_save_plot
    plot_name = args.plot_name
    render = not args.no_render
    downsample_scale = args.downsample
    steps_per_action = args.steps
    episodes = args.episodes
    epsilon = args.epsilon
    final_epsilon = args.final_epsilon
    epsilon_decay = args.decay
    total_observe_count = args.observe
    replay_memory_size = args.replay_memory
    batch_size = args.batch
    gamma = args.gamma

    # Check arguments.
    run_checks()

    # Create the skiing environment.
    env, state, pixel_rows, pixel_columns, action_space_size = create_skiing_environment()
    # Create the observation space's shape.
    observation_space_shape = (pixel_rows // downsample_scale, pixel_columns // downsample_scale, agent_frame_history)

    # Create the optimizer.
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)

    # Create the agent.
    agent = create_agent()

    # Initialize arrays for the scores.
    max_scores = np.empty(episodes)
    total_scores = np.empty(episodes)

    # Start the game loop.
    game_loop()
