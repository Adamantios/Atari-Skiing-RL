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
        self.steps_taken = 0
        self.observing = True

    def _decay_epsilon(self) -> None:
        """ Decays the policy's epsilon. """
        if self.e > self.final_e and not self.observing:
            self.e -= self.epsilon_decay

            if self.e < self.final_e:
                self.e = self.final_e

    def _update_steps(self) -> None:
        """ Updates the number of steps and sets the observing value if needed. """
        if self.steps_taken < self.total_observe_count:
            self.steps_taken += 1
            if self.total_observe_count == self.steps_taken:
                self.observing = False
                print('Agent has stopped observing.\nThings are about to get serious!\nOr not...')

    def take_action(self, model: Model, current_state: np.ndarray) -> int:
        """
        Takes an action based on the policy.

        :param model: the model to use.
        :param current_state: the state for which the action will be taken.
        :return: the action number.
        """
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
