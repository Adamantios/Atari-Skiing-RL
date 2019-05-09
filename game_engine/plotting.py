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
            fig, ax = plt.subplots(figsize=(12, 10))
            # Start from 1, not 0.
            ax.set_xlim(1, self.episodes)
            ax.plot(np.append(np.roll(score, 1), score[self.episodes - 1]))

            # Arrange ticks, only if the episodes are less or equal with 20.
            if self.episodes <= 20:
                ax.set_xticks(range(1, self.episodes + 1))
            else:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Add title and labels.
            ax.set_title(title, fontsize='x-large')
            ax.set_xlabel('Episode', fontsize='large')
            ax.set_ylabel('Score', fontsize='large')

            self._plot_and_save(fig, self.plots_name_prefix + filename_suffix)
