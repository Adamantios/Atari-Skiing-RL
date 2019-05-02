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

    def plot_score_vs_episodes(self, score: np.ndarray, title: str, filename_suffix: str) -> None:
        """
        Plots a score array vs episodes.

        :param score: the array with the scores.
        :param title: the plot's title.
        :param filename_suffix: the saved plot's filename suffix.
        """
        if self.plot_train_results or self.save_plot:
            fig = plt.figure(figsize=(12, 10))
            # Start from 1, not 0.
            plt.xlim(1, self.episodes)
            plt.plot(np.append(np.roll(score, 1), score[self.episodes - 1]))
            plt.xticks(range(1, self.episodes + 1))
            plt.title(title, fontsize='x-large')
            plt.xlabel('Episode', fontsize='large')
            plt.ylabel('Score', fontsize='large')

            self._plot_and_save(fig, self.plots_name_prefix + filename_suffix)
