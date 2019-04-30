from os import path
from typing import Union

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from keras import Model
from keras.engine.saving import load_model
from keras.optimizers import RMSprop

from agent import EGreedyPolicy, DQN
from model import atari_skiing_model, huber_loss
from utils import create_path, atari_preprocess


def run_checks() -> None:
    """ Checks the default variables. """
    # TODO add more checks.
    # Create the path to the file, if necessary.
    create_path(filename_prefix)

    if not path.exists(model_path):
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


def game_loop() -> None:
    """ Starts the game loop and trains the agent. """
    # Run for a number of episodes.
    for episode in range(nEpisodes):
        # Init vars.
        max_score, score, done = 0, 0, False

        # Reset and render the environment.
        current_state = env.reset()
        render_frame()

        for _ in range(steps_per_action):
            current_state, _, _, _ = env.step(1)
            render_frame()

        current_state = atari_preprocess(current_state, downsample_scale)
        current_state = np.stack((current_state, current_state, current_state), axis=2)
        current_state = np.reshape([current_state],
                                   (1, pixel_rows // downsample_scale, pixel_columns // downsample_scale,
                                    action_space_size))

        while not done:
            # Take an action, using the policy.
            action = policy.take_action(episode, model, current_state)
            # Take a step, using the action.
            next_state, reward, done, _ = env.step(action)
            # Render the frame.
            render_frame()

            # Preprocess the state.
            next_state = atari_preprocess(next_state, downsample_scale)
            next_state = np.append(next_state, current_state[:, :, :, :], axis=3)

            replay_memory.append((current_state, action, reward, next_state))

            if episode + 1 > total_observe_count:
                agent.fit()

                if episode % target_model_change == 0:
                    target_model.set_weights(model.get_weights())

            score += reward
            current_state = next_state

            if max_score < score:
                print("max score for the episode {} is : {} ".format(episode + 1, score))
                max_score = score

        if episode % 99 == 0:
            print("final score for the episode {} is : {} ".format(episode + 1, score))
            model.save("{}_{}.h5".format(filename_prefix, episode + 1))


if __name__ == '__main__':
    # Create the default variables.
    filename_prefix = 'out/atari_skiing'
    model_path = 'out/atari_skiing_1.h5'
    render = True
    downsample_scale = 2
    steps_per_action = 3
    nEpisodes = 1
    epsilon = 1.
    total_observe_count = 750
    batch_size = 32
    gamma = .99
    final_epsilon = .1
    epsilon_decay = 1e-4
    target_model_change = 100
    replay_memory_size = 400000

    # Check variables.
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
