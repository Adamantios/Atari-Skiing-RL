import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.optimizers import RMSprop

from agent import e_greedy_policy_action, DQN
from model import atari_skiing_model
from utils import create_path, atari_preprocess


def create_skiing_environment():
    environment = gym.make('Skiing-v0')
    init_state = environment.reset()
    height, width = environment.observation_space.shape[0], environment.observation_space.shape[1]
    act_space_size = environment.action_space.n

    return environment, init_state, height, width, act_space_size


def game_loop():
    for episode in range(nEpisodes):
        global epsilon
        max_score, score, done = 0, 0, False

        # Reset the environment.
        current_state = env.reset()
        # Render the environment.
        env.render()

        for _ in range(steps_per_action):
            current_state, _, _, _ = env.step(1)
            env.render()

        current_state = atari_preprocess(current_state, downsample_scale)
        current_state = np.stack((current_state, current_state, current_state), axis=2)
        current_state = np.reshape([current_state],
                                   (1, pixel_rows // downsample_scale, pixel_columns // downsample_scale,
                                    action_space_size))

        while not done:
            action = e_greedy_policy_action(epsilon, model, episode, total_observe_count, current_state,
                                            action_space_size)

            if epsilon > final_epsilon and episode > total_observe_count:
                epsilon -= epsilon_decay

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = atari_preprocess(next_state, downsample_scale)
            next_state = np.append(next_state, current_state[:, :, :, :5], axis=3)

            replay_memory.append((current_state, action, reward, next_state))

            if episode > total_observe_count:
                agent.fit()

                if episode % target_model_change == 0:
                    target_model.set_weights(model.get_weights())

            score += reward
            current_state = next_state

            if max_score < score:
                print("max score for the episode {} is : {} ".format(episode, score))
                max_score = score

        if episode % 100 == 0:
            print("final score for the episode {} is : {} ".format(episode + 1, score))
            model.save("{}_{}.h5".format(filename_prefix, episode + 1))


if __name__ == '__main__':
    filename_prefix = 'out/atari_skiing'
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

    create_path(filename_prefix)

    env, state, pixel_rows, pixel_columns, action_space_size = create_skiing_environment()
    observation_space_shape = (pixel_rows // downsample_scale, pixel_columns // downsample_scale, steps_per_action)

    replay_memory = deque(maxlen=replay_memory_size)

    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model = atari_skiing_model(observation_space_shape, action_space_size, optimizer)
    target_model = atari_skiing_model(observation_space_shape, action_space_size, optimizer)

    agent = DQN(model, target_model, replay_memory, gamma, batch_size, action_space_size)

    game_loop()
