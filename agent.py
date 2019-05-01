from random import randrange, sample

import numpy as np
from keras.utils import to_categorical


class DQN(object):
    def __init__(self, model, target_model, memory, gamma, batch_size, observation_space_shape, action_size):
        self.target_model = target_model
        self.model = model
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.observation_space_shape = observation_space_shape
        self.action_size = action_size

    def fit(self):
        current_state_batch, actions, rewards, next_state_batch = \
            get_batch_from_replay_memory(self.memory, self.batch_size, self.observation_space_shape)

        actions_mask = np.ones((self.batch_size, self.action_size))
        next_q_values = self.target_model.predict([next_state_batch, actions_mask])  # separate old model to predict

        targets = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            targets[i] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        one_hot_actions = to_categorical(actions, self.action_size)
        one_hot_targets = one_hot_actions * targets[:, None]

        self.model.fit([current_state_batch, one_hot_actions],
                       one_hot_targets, epochs=1, batch_size=self.batch_size, verbose=0)


class EGreedyPolicy(object):
    def __init__(self, e: float, final_e: float, epsilon_decay: float, total_observe_count: int, action_size: int):
        self.e = e
        self.final_e = final_e
        self.epsilon_decay = epsilon_decay
        self.total_observe_count = total_observe_count
        self.action_size = action_size

    def _decay_epsilon(self, episode: int) -> float:
        if self.e > self.final_e and episode > self.total_observe_count:
            self.e -= self.epsilon_decay

        return self.e

    def take_action(self, episode: int, model, current_state):
        if np.random.rand() <= self.e or episode < self.total_observe_count:
            # Take random action.
            action = randrange(self.action_size)
        else:
            # Take the best action.
            q_value = model.predict([current_state, np.expand_dims(np.ones(self.action_size), 0)])
            action = np.argmax(q_value[0])

        # Decay epsilon.
        self._decay_epsilon(episode)

        return action


def get_batch_from_replay_memory(memory, batch_size, observation_space_shape):
    mini_batch = sample(memory, batch_size)

    current_state_batch, next_state_batch = \
        np.empty((batch_size, observation_space_shape[0], observation_space_shape[1], observation_space_shape[2])), \
        np.empty((batch_size, observation_space_shape[0], observation_space_shape[1], observation_space_shape[2]))

    actions, rewards = [], []

    for idx, val in enumerate(mini_batch):
        current_state_batch[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_state_batch[idx] = val[3]

    return current_state_batch, actions, rewards, next_state_batch
