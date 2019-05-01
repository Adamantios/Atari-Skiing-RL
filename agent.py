import pickle
from collections import deque
from os import remove
from os.path import basename, splitext, dirname, join
from random import randrange, sample
from zipfile import ZipFile

import numpy as np
from keras import Model
from keras.engine.saving import load_model
from keras.models import clone_model
from keras.utils import to_categorical


class EGreedyPolicy(object):
    def __init__(self, e: float, final_e: float, epsilon_decay: float, total_observe_count: int, action_size: int):
        self.e = e
        self.final_e = final_e
        self.epsilon_decay = epsilon_decay
        self.total_observe_count = total_observe_count
        self.action_size = action_size
        self.steps_taken = 0
        self.observing = True

    def _decay_epsilon(self) -> float:
        if self.e > self.final_e and not self.observing:
            self.e -= self.epsilon_decay

        return self.e

    def _update_steps(self):
        if self.steps_taken < self.total_observe_count:
            self.steps_taken += 1
            if self.total_observe_count == self.steps_taken:
                self.observing = False
                print('Agent has stopped observing.\nThings are about to get serious!\nOr not...')

    def take_action(self, model: Model, current_state: np.ndarray) -> int:
        # Update steps.
        self._update_steps()

        if np.random.rand() <= self.e or self.observing:
            # Take random action.
            action = randrange(self.action_size)
        else:
            # Take the best action.
            q_value = model.predict([current_state, np.expand_dims(np.ones(self.action_size), 0)])
            action = np.argmax(q_value[0])

        # Decay epsilon.
        self._decay_epsilon()

        return action


class DQN(object):
    def __init__(self, model: Model, target_model_change: int, memory: deque, gamma: float, batch_size: int,
                 observation_space_shape: tuple, action_size: int, policy: EGreedyPolicy):
        self.model = model
        self.target_model_change = target_model_change
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.observation_space_shape = observation_space_shape
        self.action_size = action_size
        self.policy = policy
        self.target_model = clone_model(model)
        self.steps_from_update = 0

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

    def fit(self):
        if not self.policy.observing:
            self.steps_from_update += 1

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

            if self.steps_from_update == self.target_model_change or self.target_model_change < 1:
                print('Updating target model.')
                self.update_target_model()
                print('Target model has been successfully updated.')

    def take_action(self, current_state: np.ndarray) -> int:
        return self.policy.take_action(self.model, current_state)

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())
        self.steps_from_update = 0

    def save_agent(self, filename_prefix: str = 'dqn') -> str:
        """
        Saves the agent.

        :param filename_prefix: the agent's filename prefix.
        :return: the filename.
        """
        # Create filenames.
        model_filename = filename_prefix + '_model.h5'
        memory_filename = filename_prefix + '_memory.pickle'
        zip_filename = filename_prefix + '.zip'

        # Save model.
        self.model.save(model_filename)

        # Save memory.
        with open(memory_filename, 'wb') as stream:
            pickle.dump(self.memory, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.close()

        # Zip model and memory together.
        with ZipFile(zip_filename, 'w') as model_zip:
            model_zip.write(model_filename, basename(model_filename))
            model_zip.write(memory_filename, basename(memory_filename))
            model_zip.close()

        # Remove files out of the zip.
        remove(model_filename)
        remove(memory_filename)

        return zip_filename

    @staticmethod
    def load_agent(filename: str, custom_objects: dict) -> [Model, deque]:
        """
        Loads the agent.

        :param filename: the agent's filename.
        :param custom_objects: custom_objects for the keras model.
        """
        # Create filenames.
        directory = dirname(filename)
        basename_no_extension = basename(splitext(filename)[0])
        model_filename = join(directory, basename_no_extension + '_model.h5')
        memory_filename = join(directory, basename_no_extension + '_memory.pickle')

        # Read model and memory.
        with ZipFile(filename) as model_zip:
            model_zip.extractall(directory)
            model_zip.close()

        # Load model.
        model = load_model(model_filename, custom_objects=custom_objects)

        # Load memory.
        with open(memory_filename, 'rb') as stream:
            memory = pickle.load(stream)
            stream.close()

        # Remove files out of the zip.
        remove(model_filename)
        remove(memory_filename)

        return model, memory

    def append_to_memory(self, current_state, action, reward, next_state):
        self.memory.append((current_state, action, reward, next_state))
