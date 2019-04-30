from random import randrange, sample

import numpy as np


class DQN(object):
    def __init__(self, model, target_model, memory, gamma, batch_size, action_size):
        self.target_model = target_model
        self.model = model
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_size = action_size

    def fit(self):
        current_state_batch, actions, rewards, next_state_batch = get_batch_from_replay_memory(self.batch_size,
                                                                                               self.memory)

        actions_mask = np.ones((self.batch_size, self.action_size))
        next_q_values = self.target_model.predict([next_state_batch, actions_mask])  # separate old model to predict

        targets = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            targets[i] = rewards[i] + self.gamma * np.amax(next_q_values[i])

        one_hot_actions = np.eye(self.action_size)[np.array(actions).reshape(-1)]
        one_hot_targets = one_hot_actions * targets[:, None]

        self.model.fit([current_state_batch, one_hot_actions],
                       one_hot_targets, epochs=1, batch_size=self.batch_size, verbose=0)


def e_greedy_policy_action(e, model, episode, total_observe_count, current_state, action_size):
    if np.random.rand() <= e or episode < total_observe_count:
        # Take random action.
        return randrange(action_size)
    else:
        # Take the best action.
        q_value = model.predict([current_state, np.ones(action_size).reshape(1, action_size)])
        return np.argmax(q_value[0])


def get_batch_from_replay_memory(memory, batch_size):
    mini_batch = sample(memory, batch_size)

    current_state_batch = np.zeros((batch_size, 84, 84, 4))
    next_state_batch = np.zeros((batch_size, 84, 84, 4))

    actions, rewards = [], []

    for idx, val in enumerate(mini_batch):
        current_state_batch[idx] = val[0]
        actions.append(val[1])
        rewards.append(val[2])
        next_state_batch[idx] = val[3]

    return current_state_batch, actions, rewards, next_state_batch
