from collections import deque
from random import randrange, sample

import numpy as np
from keras import Model
from keras.models import clone_model
from keras.utils import to_categorical


class DQN(object):
    def __init__(self, model: Model, target_model_change: int, memory: deque, gamma: float, batch_size: int,
                 observation_space_shape: tuple, action_size: int, save_interval: int, filename_prefix: str, policy):
        self.model = model
        self.target_model_change = target_model_change
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.observation_space_shape = observation_space_shape
        self.action_size = action_size
        self.save_interval = save_interval
        self.filename_prefix = filename_prefix
        self.policy = policy
        self.target_model = clone_model(model)

    def _get_mini_batch(self) -> [np.ndarray]:
        """
        Samples a random mini batch from the replay memory.

        :return: the current state batch, the actions batch, the rewards batch and the next state batch
        """
        # Randomly sample a mini batch.
        mini_batch = sample(self.memory, self.batch_size)

        # Initialize arrays.
        current_state_batch, next_state_batch, actions, rewards = \
            np.empty(((self.batch_size,) + self.observation_space_shape)), \
            np.empty(((self.batch_size,) + self.observation_space_shape)), \
            np.empty(self.batch_size), \
            np.empty(self.batch_size)

        # Get values from the mini batch.
        for idx, val in enumerate(mini_batch):
            current_state_batch[idx] = val[0]
            actions[idx] = val[1]
            rewards[idx] = val[2]
            next_state_batch[idx] = val[3]

        return current_state_batch, actions, rewards, next_state_batch

    def fit(self, episode: int):
        current_state_batch, actions, rewards, next_state_batch = self._get_mini_batch()

        actions_mask = np.ones((self.batch_size, self.action_size))
        next_q_values = self.target_model.predict([next_state_batch, actions_mask])  # separate old model to predict

        targets = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            targets[i] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        one_hot_actions = to_categorical(actions, self.action_size)
        one_hot_targets = one_hot_actions * targets[:, None]

        self.model.fit([current_state_batch, one_hot_actions],
                       one_hot_targets, epochs=1, batch_size=self.batch_size, verbose=0)

        if episode % self.target_model_change == 0 or self.target_model_change < 2:
            self.update_target_model()

        if episode % self.save_interval == 0 or self.save_interval < 2:
            self.save_model("{}_{}.h5".format(self.filename_prefix, episode))

    def take_action(self, current_state: np.ndarray, episode: int) -> int:
        return self.policy.take_action(self.model, current_state, episode)

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filename: str) -> None:
        """
        Saves the model after an episode.

        :param filename: the model's filename.
        """
        self.model.save(filename)


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

    def take_action(self, model: Model, current_state: np.ndarray, episode: int) -> int:
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
