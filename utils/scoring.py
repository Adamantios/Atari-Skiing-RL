import pickle

import numpy as np


class Scorer(object):
    def __init__(self, episodes: int, info_interval_mean: int, results_name_prefix: str):
        self.episodes = episodes
        self.info_interval_mean = info_interval_mean
        self.results_name_prefix = results_name_prefix

        # Initialize arrays for the scores and history.
        self.max_scores = np.empty(episodes, dtype=np.float)
        self.total_scores = np.empty(episodes, dtype=np.float)
        self.huber_loss_history = np.zeros(episodes, dtype=np.float)

    def save_results(self, episode: int) -> None:
        """
        Saves the results in files.

        :param episode: the current episode.
        """
        print('Saving results.')

        # Save total scores.
        with open('{}_total_scores_episode{}'.format(self.results_name_prefix, episode), 'wb') as stream:
            pickle.dump(self.total_scores, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.close()

        # Save max scores.
        with open('{}_max_scores_episode{}'.format(self.results_name_prefix, episode), 'wb') as stream:
            pickle.dump(self.max_scores, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.close()

        # Save losses.
        with open('{}_losses_episode{}'.format(self.results_name_prefix, episode), 'wb') as stream:
            pickle.dump(self.huber_loss_history, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.close()

        print('Results have been saved successfully.')

    def show_episode_scoring(self, episode: int) -> None:
        """
        Shows an episode's scoring information.

        :param episode: the episode for which the actions will be taken.
        """
        # Print the episode's scores.
        print("Max score for the episode {} is: {} ".format(episode, self.max_scores[episode - 1]))
        print("Total score for the episode {} is: {} ".format(episode, self.total_scores[episode - 1]))

    def show_mean_scoring(self, episode: int) -> None:
        """
        Shows the mean scoring information for the current episode.

        :param episode: the episode.
        """
        # Print the episodes mean scores.
        mean_max_score = \
            self.max_scores[episode - self.info_interval_mean:episode].sum() / self.info_interval_mean
        mean_total_score = \
            self.total_scores[episode - self.info_interval_mean:episode].sum() / self.info_interval_mean

        print("Mean Max score for {}-{} episodes is: {} "
              .format(episode - self.info_interval_mean, episode, mean_max_score))
        print("Mean Total score for {}-{} episodes is: {} "
              .format(episode - self.info_interval_mean, episode, mean_total_score))
