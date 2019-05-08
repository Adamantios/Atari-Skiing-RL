from collections import Generator
from dataclasses import dataclass
from random import randint

import gym
import numpy as np
from gym.wrappers import TimeLimit
from math import inf, ceil

from core.agent import DQN
from utils.os_operations import print_progressbar
from utils.plotting import Plotter
from utils.preprocessing import atari_preprocess
from utils.scoring import Scorer


@dataclass
class GameResultSpecs:
    info_interval_current: int
    info_interval_mean: int
    agent_save_interval: int
    results_save_interval: int
    plots_name_prefix: str = 'plots'
    results_name_prefix: str = 'results'
    agent_name_prefix: str = 'agent'
    plot_train_results: bool = True
    save_plots: bool = True


_GameInfo = [np.ndarray, float, bool]


class Game(object):
    def __init__(self, episodes: int, downsample_scale: int, agent_frame_history: int, steps_per_action: int,
                 fit_frequency: int, no_operation: int, specs: GameResultSpecs):
        self.episodes = episodes
        self.downsample_scale = downsample_scale
        self.agent_frame_history = agent_frame_history
        self.steps_per_action = steps_per_action
        self.fit_frequency = fit_frequency
        self.no_operation = no_operation
        self.specs = specs

        # Create the skiing environment.
        self._env, self.pixel_rows, self.pixel_columns, self.action_space_size = self._create_skiing_environment()

        # Create the observation space's shape.
        self.observation_space_shape = (ceil(self.pixel_rows / self.downsample_scale),
                                        ceil(self.pixel_columns / self.downsample_scale),
                                        self.agent_frame_history)

        # Create a scorer.
        self.scorer = Scorer(episodes, self.specs.info_interval_mean, self.specs.results_name_prefix)

        # Create a plotter.
        self.plotter = Plotter(self.episodes, self.specs.plots_name_prefix, self.specs.plot_train_results,
                               self.specs.save_plots)

    @staticmethod
    def _create_skiing_environment() -> [TimeLimit, int, int, int]:
        """
        Creates a skiing environment.

        :return: the skiing environment, the image's height, the image's width and the action space's size.
        """
        # Create the skiing environment.
        environment = gym.make('SkiingDeterministic-v4')
        # Get the observation space's height and width.
        height, width = environment.observation_space.shape[0], environment.observation_space.shape[1]
        # Get the number of possible moves.
        act_space_size = environment.action_space.n

        return environment, height, width, act_space_size

    def _begin_episode(self, render: bool) -> np.ndarray:
        """
        Begins an episode.

        :param render: if the initial state should render.
        :return: the initial state.
        """
        # Reset and render the environment.
        init_state = self._env.reset()
        if render:
            self._env.render()

        return init_state

    def _observe(self, init_state: np.ndarray, render: bool) -> [np.ndarray, bool]:
        """
        Take no action.

        :param init_state: the initial state.
        :param render: if the observing steps should render.
        :return: the next state and if game is done.
        """
        # Init variables.
        observe, done = init_state, False

        # Observe for a random number of steps picked from [1, self._no_operation].
        for _ in range(randint(1, self.no_operation)):
            # Take no action.
            observe, _, done, _ = self._env.step(0)
            # Render the frame.
            if render:
                self._env.render()

            if done:
                break

        return observe, done

    def _pretrain_phase(self, render: bool) -> np.ndarray:
        """
        Completes pretrain phase.

        :param render: if the pretrain phase should render.
        :return: the state that was created.
        """
        # Start the episode.
        init_state = self._begin_episode(render)

        # Just observe.
        current_state, done = self._observe(init_state, render)

        # Preprocess current_state.
        current_state = atari_preprocess(current_state, self.downsample_scale)

        # Create preceding frames, using the starting frame.
        current_state = np.stack(tuple([current_state for _ in range(self.agent_frame_history)]), axis=2)

        # Reshape the state.
        current_state = np.reshape(current_state,
                                   (1,
                                    ceil(self.pixel_rows / self.downsample_scale),
                                    ceil(self.pixel_columns / self.downsample_scale),
                                    self.agent_frame_history))

        return current_state

    def _repeat_action(self, agent: DQN, current_state: np.ndarray, action: int, render: bool) -> _GameInfo:
        """
        Repeats a game action.

        :param agent: the agent to repeat the action.
        :param current_state: the current state.
        :param action: the action to repeat.
        :param render: if the action should render.
        :return: the next state, the reward and if game is done.
        """
        # Init variables.
        reward, next_state, done = 0, current_state, False

        for _ in range(self.steps_per_action):
            # Take a step, using the action.
            next_state, new_reward, done, _ = self._env.step(action)
            # Render the frame.
            if render:
                self._env.render()
            # Add reward.
            reward += new_reward

            if done:
                break

            # Preprocess the state.
            next_state = atari_preprocess(next_state, self.downsample_scale)
            # Append the frame history.
            next_state = np.append(next_state, current_state[:, :, :, :self.agent_frame_history - 1], axis=3)

            # Save sample <s,a,r,s'> to the replay memory.
            agent.append_to_memory(current_state, action, reward, next_state)

            # Set current state with the next.
            current_state = next_state

        return next_state, reward, done

    def _take_action(self, agent: DQN, current_state: np.ndarray, episode: int, render: bool) -> _GameInfo:
        """
        Takes an action.

        :param agent: the agent to take the action.
        :param current_state: the current state.
        :param episode: the current episode.
        :param render: if the action should render.
        :return: the next state, the reward and if game is done.
        """
        # Take an action, using the policy.
        action = agent.take_action(current_state, episode)
        # Repeat the action.
        next_state, reward, done = self._repeat_action(agent, current_state, action, render)

        return next_state, reward, done

    def _train_and_play(self, agent: DQN, current_state: np.ndarray, episode: int, render: bool) -> _GameInfo:
        """
        Train the agent while playing, using the current state.

        :param agent: the agent to train.
        :param current_state: the current state.
        :param episode: the current episode.
        :param render: if the actions should render.
        :return: the next state, the reward and if game is done.
        """
        # Init variables.
        reward, next_state, done = 0, current_state, False

        # Repeat actions before fitting time.
        for _ in range(self.fit_frequency):
            # Take an action.
            next_state, new_reward, done = self._take_action(agent, current_state, episode, render)
            # Add reward.
            reward += new_reward
            if done:
                break

        # Fit agent and keep fitting history.
        fitting_history = agent.fit()
        if fitting_history is not None:
            self.scorer.huber_loss_history[episode - 1] += fitting_history.history['loss']

        return next_state, reward, done

    def _game_loop(self, agent: DQN, render: bool) -> Generator:
        """
        Starts the game loop and trains the agent.

        :param agent: the agent to play the game.
        :param render: if the game should render.
        :return: generator containing the finished episode number.
        """
        # Run for a number of episodes.
        for episode in range(1, self.episodes + 1):
            # Init vars.
            reward, max_score, total_score, done = 0, -inf, 0, False
            # Complete pretrain phase.
            current_state = self._pretrain_phase(render)

            while not done:
                # Train the agent while playing.
                current_state, reward, done = self._train_and_play(agent, current_state, episode, render)
                # Add reward to the total score.
                total_score += reward
                # Set max score.
                max_score = max(max_score, reward)

            # Add scores to the scores arrays.
            self.scorer.max_scores[episode - 1] = max_score
            self.scorer.total_scores[episode - 1] = total_score

            # Yield the finished episode.
            yield episode

    def _update_progressbar(self, finished_episode: int) -> None:
        """
        Updates game progressbar.

        :param finished_episode: the episode that just finished.
        """
        if not self.specs.info_interval_current == 1 and finished_episode != self.episodes:
            # Reinitialize progressbar if it just finished, but the game did not.
            if finished_episode % self.specs.info_interval_current == 0:
                print_progressbar(0, self.specs.info_interval_current,
                                  'Episode: 0/{}'.format(self.specs.info_interval_current),
                                  'Finished: {}/{}'.format(finished_episode, self.episodes))

            else:
                print_progressbar(finished_episode % self.specs.info_interval_current, self.specs.info_interval_current,
                                  'Episode: {}/{}'.format(finished_episode % self.specs.info_interval_current,
                                                          self.specs.info_interval_current),
                                  'Finished: {}/{}'.format(finished_episode, self.episodes))

    def _end_of_episode_actions(self, finished_episode: int, agent: DQN) -> None:
        """
        Takes actions after the episode finishes.
        Shows scoring information and saves the model.

        :param finished_episode: the episode for which the actions will be taken.
        :param agent: the agent with whom the episode was played.
        """
        # Save agent.
        if finished_episode % self.specs.agent_save_interval == 0 or self.specs.agent_save_interval == 1:
            print('Saving agent.')
            filename = agent.save_agent("{}_{}".format(self.specs.agent_name_prefix, finished_episode))
            print('Agent has been successfully saved as {}.'.format(filename))

        # Show scores.
        if finished_episode % self.specs.info_interval_current == 0 or self.specs.info_interval_current == 1:
            self.scorer.show_episode_scoring(finished_episode)

        if self.specs.info_interval_mean > 1 and finished_episode % self.specs.info_interval_mean == 0:
            self.scorer.show_mean_scoring(finished_episode)

        # Update progressbar.
        self._update_progressbar(finished_episode)

        # Plot scores.
        if finished_episode == self.episodes and self.episodes > 1:
            # Max score.
            self.plotter.plot_score_vs_episodes(self.scorer.max_scores, 'Max Score vs Episodes',
                                                '_max_scores_vs_episodes.png')
            # Total score.
            self.plotter.plot_score_vs_episodes(self.scorer.total_scores, 'Total Score vs Episodes',
                                                '_total_scores_vs_episodes.png')
            # Huber loss.
            self.plotter.plot_score_vs_episodes(self.scorer.huber_loss_history,
                                                'Total Huber loss vs episodes', '_loss_vs_episodes.png')

        # Save results.
        if self.specs.results_save_interval > 0 and (
                finished_episode % self.specs.results_save_interval == 0 or self.specs.results_save_interval == 1):
            self.scorer.save_results(finished_episode)

    def play_game(self, agent: DQN, render: bool) -> None:
        """
        Plays the game.

        :param agent: the agent who will take the actions in the game.
        :param render: if the game should render.
        """
        # Initialize progressbar.
        self._update_progressbar(0)

        # Start the game loop.
        for finished_episode in self._game_loop(agent, render):
            # Take specific actions after the end of each episode.
            self._end_of_episode_actions(finished_episode, agent)
