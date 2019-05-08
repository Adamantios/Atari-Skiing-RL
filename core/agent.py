import pickle
from collections import deque
from os import remove
from os.path import basename, splitext, dirname, join
from random import sample
from zipfile import ZipFile

import numpy as np
from keras import Model
from keras.callbacks import History
from keras.engine.saving import load_model
from keras.models import clone_model
from keras.utils import to_categorical

from core.policy import EGreedyPolicy


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

    def take_action(self, current_state: np.ndarray, episode: int) -> int:
        """
        Takes an action based on the policy.

        :param current_state: the state for which the action will be taken.
        :param episode: the current episode.
        :return: the action number.
        """
        return self.policy.take_action(self.model, current_state, episode)

    def append_to_memory(self, current_state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        """
        Adds values to the agent's memory.

        :param current_state: the state to add.
        :param action: the action to add.
        :param reward: the reward to add.
        :param next_state: the next state to add.
        """
        self.memory.append((current_state, action, reward, next_state))

    def update_target_model(self) -> None:
        """ Updates the target model. """
        self.target_model.set_weights(self.model.get_weights())
        self.steps_from_update = 0

    def fit(self) -> History:
        """
        Fits the agent.

        :return: the fit history.
        """
        # Fit only if the agent is not observing.
        if not self.policy.observing:
            # Increase the steps from update indicator.
            self.steps_from_update += 1

            # Get the mini batches.
            current_state_batch, actions, rewards, next_state_batch = self._get_mini_batch()

            # Create the actions mask.
            actions_mask = np.ones((self.batch_size, self.action_size))
            # Predict the next QValues.
            next_q_values = self.target_model.predict([next_state_batch, actions_mask])
            # Initialize the target QValues for the mini batch.
            target_q_values = np.empty((self.batch_size,))

            for i in range(self.batch_size):
                # Update rewards, using the Deep Q Learning rule.
                target_q_values[i] = rewards[i] + self.gamma * np.amax(next_q_values[i])

            # One hot encode the actions.
            one_hot_actions = to_categorical(actions, self.action_size)
            # One hot encode the target QValues.
            one_hot_target_q_values = one_hot_actions * np.expand_dims(target_q_values, 1)

            # Fit the model to the batches.
            history = self.model.fit([current_state_batch, one_hot_actions],
                                     one_hot_target_q_values, epochs=1, batch_size=self.batch_size, verbose=0)

            # Update the target model if necessary.
            if self.steps_from_update == self.target_model_change or self.target_model_change < 1:
                print('Updating target model.')
                self.update_target_model()
                print('Target model has been successfully updated.')

            return history

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
        Loads an agent from a file.

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
