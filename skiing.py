from os import path
import gym
from collections import deque
from keras import Model
from keras.engine.saving import load_model
from keras.optimizers import RMSprop
from math import inf

from agent import EGreedyPolicy, DQN
from model import atari_skiing_model, huber_loss
from utils import create_path, atari_preprocess, create_parser


def run_checks() -> None:
    """ Checks the default variables. """
    # TODO add more checks.
    # Create the path to the file, if necessary.
    create_path(filename_prefix)

    if not path.exists(model_path) and model_path != '':
        raise FileNotFoundError('File {} not found.'.format(model_path))

    if batch_size > total_observe_count:
        raise ValueError('Batch size({}) should be less than total_observe_count({}).'
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


def create_models() -> [Model, Model]:
    """
    Creates the model and the target model.

    :return: the models.
    """
    # Init the target model.
    target = atari_skiing_model(observation_space_shape, action_space_size, optimizer)

    if model_path != '':
        # Load the model.
        learning = load_model(model_path, custom_objects={'huber_loss': huber_loss})
        # Set the target model's weights with the model's.
        target.set_weights(learning.get_weights())

    else:
        # Init the model.
        learning = atari_skiing_model(observation_space_shape, action_space_size, optimizer)

    return learning, target


def render_frame() -> None:
    """ Renders a frame, only if the user has chosen to do so. """
    if render:
        env.render()


def show_episode_scoring(episode: int, max_score: int, total_score: int) -> None:
    """
    Shows episode's scoring information.

    :param episode: the episode.
    :param max_score: the max score for this episode.
    :param total_score: the episode's score.
    """
    if episode % info_interval == 0 or info_interval < 2:
        # Print the episode's scores.
        print("Max score for the episode {} is: {} ".format(episode, max_score))
        print("Total score for the episode {} is: {} ".format(episode, total_score))


def save_model(episode: int) -> None:
    """
    Saves the model after an episode.

    :param episode: the episode.
    """
    if episode % save_interval == 0 or save_interval < 2:
        model.save("{}_{}.h5".format(filename_prefix, episode))


def end_of_episode_actions(episode: int, max_score: int, total_score: int) -> None:
    """
    Take actions after the episode finishes.

    Show scoring information and saves the model.

    :param episode: the episode for which the actions will be taken.
    :param max_score: the max score for this episode.
    :param total_score: the total score for this episode.
    """
    show_episode_scoring(episode, max_score, total_score)
    save_model(episode)


def game_loop() -> None:
    """ Starts the game loop and trains the agent. """
    # Run for a number of episodes.
    for episode in range(1, episodes + 1):
        # Init vars.
        max_score, total_score, done = -inf, 0, False

        # Reset and render the environment.
        current_state = env.reset()
        render_frame()

        # for _ in range(steps_per_action):
        #     current_state, _, _, _ = env.step(1)
        #     render_frame()
        #
        # current_state = atari_preprocess(current_state, downsample_scale)
        # current_state = np.stack((current_state, current_state, current_state), axis=2)
        # current_state = np.reshape([current_state],
        #                            (1, pixel_rows // downsample_scale, pixel_columns // downsample_scale,
        #                             action_space_size))

        while not done:
            # Take an action, using the policy.
            action = policy.take_action(episode, model, current_state)
            # Take a step, using the action.
            next_state, reward, done, _ = env.step(action)
            # Render the frame.
            render_frame()

            # Preprocess the state.
            next_state = atari_preprocess(next_state, downsample_scale)

            # next_state = np.append(next_state, current_state[:, :, :, :], axis=3)

            # Save sample <s,a,r,s'> to the replay memory.
            replay_memory.append((current_state, action, reward, next_state))

            if episode > total_observe_count:
                agent.fit()

                if episode % target_model_change == 0 or target_model_change < 2:
                    target_model.set_weights(model.get_weights())

            # Add reward to the total score.
            total_score += reward
            # Set current state with the next.
            current_state = next_state
            # Set max score.
            max_score = max(max_score, reward)

        # Take end of episode specific actions.
        end_of_episode_actions(episode, max_score, total_score)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    filename_prefix = args.filename
    save_interval = args.save_interval
    info_interval = args.info_interval
    target_model_change = args.target_interval
    model_path = args.model
    plot_train_results = not args.no_plot
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
    observation_space_shape = (pixel_rows // downsample_scale, pixel_columns // downsample_scale, steps_per_action)

    # Create the replay memory for the agent.
    replay_memory = deque(maxlen=replay_memory_size)

    # Create the optimizer.
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)

    # Create the models.
    model, target_model = create_models()

    # Create the policy.
    policy = EGreedyPolicy(epsilon, final_epsilon, epsilon_decay, total_observe_count, action_space_size)

    # Create the agent.
    agent = DQN(model, target_model, replay_memory, gamma, batch_size, observation_space_shape, action_space_size)

    # Start the game loop.
    game_loop()
