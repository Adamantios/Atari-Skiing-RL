import pickle
from os import remove
from os.path import basename, splitext, dirname, join
from random import sample
from typing import Union
from zipfile import ZipFile

import numpy as np
from keras import Model
from keras.callbacks import History
from keras.engine.saving import load_model
from keras.models import clone_model
from keras.utils import to_categorical

from core.policy import EGreedyPolicy


class ExperienceReplayMemory(object):
    """ Implements a Ring Buffer with an extra function which randomly samples elements from it,
    in order to be used as an Experience Replay Memory for the agent. """

    def __init__(self, size: int):
        # Check size value.
        if size < 1:
            raise ValueError('Memory size must be a positive integer. Got {} instead.'.format(size))

        # Initialize array.
        # Allocate one extra element, so that self.start == self.end always means the buffer is EMPTY,
        # whereas if exactly the right number of elements is allocated,
        # it also means the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        # Initialize start pointer.
        self.start = 0
        # Initialize end pointer.
        self.end = 0

    def append(self, element) -> None:
        """
        Appends an element to the memory.

        :param element: the element to append.
        """
        # Add the element to the end of the memory.
        self.data[self.end] = element
        # Increment the end pointer.
        self.end = (self.end + 1) % len(self.data)

        # Remove the first element by incrementing start pointer, if the memory size has been reached.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def randomly_sample(self, num_items: int) -> list:
        """
        Samples a number of items from the memory randomly.

        :param num_items: the number of the random items to be sampled.
        :return: the items.
        """
        # Sample a random number of memory indexes, which result in non empty contents and return the contents.
        indexes = sample(range(len(self)), num_items)
        return [self[idx] for idx in indexes]

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DQN(object):
    def __init__(self, model: Model, target_model_change: int, gamma: float, batch_size: int,
                 observation_space_shape: tuple, action_size: int, policy: EGreedyPolicy, target_model: Model = None,
                 memory_size: int = None, memory: ExperienceReplayMemory = None):
        self.model = model
        self.target_model_change = target_model_change
        self.memory = ExperienceReplayMemory(memory_size) if memory is None else memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.observation_space_shape = observation_space_shape
        self.action_size = action_size
        self.policy = policy
        self.target_model = self._create_target_model() if target_model is None else target_model
        self.steps_from_update = 0

    def _create_target_model(self) -> Model:
        """
        Creates the target model, by copying the model.

        :return: the target model.
        """
        target_model = clone_model(self.model)
        target_model.build(self.observation_space_shape)
        target_model.compile(optimizer=self.model.optimizer, loss=self.model.loss)
        target_model.set_weights(self.model.get_weights())

        return target_model

    def _get_mini_batch(self) -> [np.ndarray]:
        """
        Samples a random mini batch from the replay memory.

        :return: the current state batch, the actions batch, the rewards batch and the next state batch
        """
        # Randomly sample a mini batch.
        mini_batch = self.memory.randomly_sample(self.batch_size)

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

    def fit(self) -> Union[History, None]:
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
        model_filename = 'model.h5'
        target_model_filename = 'target_model.h5'
        config_filename = 'config.pickle'
        zip_filename = filename_prefix + '.zip'

        # Save models.
        self.model.save(model_filename)
        self.target_model.save(target_model_filename)

        # Create configuration dict.
        config = dict({
            'target_model_change': self.target_model_change,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'observation_space_shape': self.observation_space_shape,
            'action_size': self.action_size,
            'policy': self.policy,
            'memory': self.memory
        })

        # Save configuration.
        with open(config_filename, 'wb') as stream:
            pickle.dump(config, stream, protocol=pickle.HIGHEST_PROTOCOL)

        # Zip models and configuration together.
        with ZipFile(zip_filename, 'w') as model_zip:
            model_zip.write(model_filename, basename(model_filename))
            model_zip.write(target_model_filename, basename(target_model_filename))
            model_zip.write(config_filename, basename(config_filename))

        # Remove files out of the zip.
        remove(model_filename)
        remove(target_model_filename)
        remove(config_filename)

        return zip_filename


def load_dqn_agent(filename: str, custom_objects: dict) -> DQN:
    """
    Loads an agent from a file, using the given parameters.

    :param filename: the agent's filename.
    :param custom_objects: custom_objects for the keras model.
    :return: the DQN agent.
    """
    # Create filenames.
    directory = dirname(filename)
    basename_no_extension = basename(splitext(filename)[0])
    model_filename = join(directory, 'model.h5')
    target_model_filename = join(directory, 'target_model.h5')
    config_filename = join(directory, 'config.pickle')

    # Read models and memory.
    with ZipFile(filename) as model_zip:
        model_zip.extractall(directory)

    # Load models.
    model = load_model(model_filename, custom_objects=custom_objects)
    target_model = load_model(target_model_filename, custom_objects=custom_objects)

    # Load configuration.
    with open(config_filename, 'rb') as stream:
        config = pickle.load(stream)

    # Remove files out of the zip.
    remove(model_filename)
    remove(target_model_filename)
    remove(config_filename)

    return DQN(model, config['target_model_change'], config['gamma'], config['batch_size'],
               config['observation_space_shape'], config['action_size'], config['policy'], target_model,
               memory=config['memory'])
