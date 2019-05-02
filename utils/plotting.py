import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Plotter(object):
    def __init__(self, episodes: int, plots_name_prefix: str, plot_train_results: bool, save_plot: bool):
        self.episodes = episodes
        self.plots_name_prefix = plots_name_prefix
        self.plot_train_results = plot_train_results
        self.save_plot = save_plot

    def _plot_and_save(self, fig: Figure, filename: str) -> None:
        """
        Plots and saves the results, after checking if it should.

        :param fig: the figure.
        :param filename: the filename.
        """
        if self.plot_train_results:
            plt.show()

        if self.save_plot:
            fig.savefig(filename)

    def plot_scores_vs_episodes(self, max_scores: np.ndarray, total_scores: np.ndarray) -> None:
        """
        Plots scores vs episodes.

        :param max_scores: the max scores array.
        :param total_scores: the total scores array.
        """
        if self.plot_train_results or self.save_plot:
            fig = plt.figure(figsize=(12, 10))
            # Start from 1, not 0.
            plt.xlim(1, self.episodes)
            plt.plot(np.append(np.roll(max_scores, 1), max_scores[self.episodes - 1]))
            plt.plot(np.append(np.roll(total_scores, 1), total_scores[self.episodes - 1]))
            plt.xticks(range(1, self.episodes + 1))
            plt.title('Scores vs Episodes', fontsize='x-large')
            plt.xlabel('Episode', fontsize='large')
            plt.ylabel('Score', fontsize='large')
            plt.legend(['Max Score', 'Total Score'], loc='upper left', fontsize='large')

            self._plot_and_save(fig, self.plots_name_prefix + '_scores_vs_episodes.png')

    def plot_loss_vs_episodes(self, huber_loss_history: np.ndarray) -> None:
        """
        Plots huber loss vs episodes.

        :param huber_loss_history: the huber loss history array.
        """
        if self.plot_train_results or self.save_plot:
            fig = plt.figure(figsize=(12, 10))
            # Start from 1, not 0.
            plt.xlim(1, self.episodes)
            plt.plot(np.append(np.roll(huber_loss_history, 1), huber_loss_history[self.episodes - 1]))
            plt.xticks(range(1, self.episodes + 1))
            plt.title('Total Huber loss vs episodes', fontsize='x-large')
            plt.xlabel('Episode', fontsize='large')
            plt.ylabel('Loss', fontsize='large')
            plt.legend(['train', 'test'], loc='upper left', fontsize='large')

            self._plot_and_save(fig, self.plots_name_prefix + '_loss_vs_episodes.png')
