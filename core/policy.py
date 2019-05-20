from random import randrange

import numpy as np
from keras import Model


class EGreedyPolicy(object):
    def __init__(self, e: float, final_e: float, epsilon_decay: float, total_observe_count: int, action_size: int):
        self.e = e
        self.final_e = final_e
        self.epsilon_decay = epsilon_decay
        self.total_observe_count = total_observe_count
        self.action_size = action_size
        self.observing_steps_taken = 0
        self.observing = False if self.total_observe_count == 0 else True
        self.episode_observation_stopped = 0

    def _decay_epsilon(self, episode: int) -> None:
        """
        Decays the policy's epsilon.

        :param episode: the current episode.
        """
        if self.e > self.final_e and not self.observing:
            self.e -= self.epsilon_decay

            if self.e < self.final_e:
                self.e = self.final_e

            if self.e == self.final_e:
                print('Final epsilon reached at episode {}'.format(episode))

    def _update_observation_state(self, episode: int) -> None:
        """
        Updates the observing value if needed.

        :param episode: the current episode.
        """
        if self.observing:
            self.observing_steps_taken += 1
            if self.total_observe_count == self.observing_steps_taken:
                self.observing = False
                self.episode_observation_stopped = episode
                print('Agent has stopped observing at episode {}.\nThings are about to get serious!\nOr not...'
                      .format(self.episode_observation_stopped))

    def take_action(self, model: Model, current_state: np.ndarray, episode: int) -> int:
        """
        Takes an action based on the policy.

        :param model: the model to use.
        :param current_state: the state for which the action will be taken.
        :param episode: the current episode.
        :return: the action number.
        """
        if np.random.rand() <= self.e or self.observing:
            # Take random action.
            action = randrange(self.action_size)
        else:
            # Take the best action.
            q_value = model.predict([current_state, np.expand_dims(np.ones(self.action_size), 0)])
            action = np.argmax(q_value[0])

        # Decay epsilon.
        self._decay_epsilon(episode)

        # Update observation state.
        self._update_observation_state(episode)

        return action
