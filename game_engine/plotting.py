import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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

    def _plot_score_vs_episodes(self, score: np.ndarray, title: str, y_label: str) -> [Figure, Axes]:
        """
        Plots a score array vs episodes.

        :param score: the array with the scores.
        :param title: the plot's title.
        :param y_label: the y label text.
        """
        if self.plot_train_results or self.save_plot:
            # Create subplot.
            fig, ax = plt.subplots(figsize=(12, 10))

            # Set x and y.
            x = np.asarray(range(self.episodes + 1))
            y = np.append(np.roll(score, 1), score[self.episodes - 1])

            # Plot mean.
            mean = np.mean(score)
            ax.plot(np.asarray([mean for _ in x]), label='Mean={}'.format(mean))

            # Plot data.
            ax.set_xlim(1, self.episodes)
            ax.plot(y, label='Score')

            # Unroll x and y.
            x = x[1:]
            y = y[1:]

            # Calculate max and min values.
            x_max, y_max = x[np.argmax(y)], y.max()
            x_min, y_min = x[np.argmin(y)], y.min()

            # Annotate max and min values, only if they are different.
            if x_max != x_min:
                ax.scatter(x_max, y_max, label='Episode={}, Max Score={}'.format(x_max, y_max),
                           color='#161925', s=150, marker='*')
                ax.scatter(x_min, y_min, label='Episode={}, Min Score={}'.format(x_min, y_min),
                           color='#f1d302', s=150, marker='X')

            # Arrange ticks, only if the episodes are less or equal with 20.
            if self.episodes <= 20:
                ax.set_xticks(range(1, self.episodes + 1))
            else:
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Add title, legends and labels.
            ax.set_title(title, fontsize='x-large')
            ax.legend()
            ax.set_xlabel('Episode', fontsize='large')
            ax.set_ylabel(y_label, fontsize='large')

            return fig, ax

    def plot_max_score_vs_episodes(self, score: np.ndarray, title: str, filename_suffix: str) -> None:
        """
        Plots max score array vs episodes.

        :param score: the array with the scores.
        :param title: the plot's title.
        :param filename_suffix: the saved plot's filename suffix.
        """
        fig, ax = self._plot_score_vs_episodes(score, title, 'Score')
        self._plot_and_save(fig, self.plots_name_prefix + filename_suffix)

    def plot_total_score_vs_episodes(self, score: np.ndarray, title: str, filename_suffix: str,
                                     compare: bool = True) -> None:
        """
        Plots total score array vs episodes.

        :param score: the array with the scores.
        :param title: the plot's title.
        :param filename_suffix: the saved plot's filename suffix.
        :param compare: whether the plot should be compared with the state of art.
        """
        fig, ax = self._plot_score_vs_episodes(score, title, 'Score')

        if compare:
            # TODO add more state of the art measurements.
            random = -17098.1
            ax.plot(np.asarray([random for _ in range(self.episodes + 1)]), label='Random={}'.format(random),
                    color='green')

            # Reset legends.
            ax.legend()
        self._plot_and_save(fig, self.plots_name_prefix + filename_suffix)

    def plot_huber_loss_vs_episodes(self, loss: np.ndarray, title: str, filename_suffix: str) -> None:
        """
        Plots max score array vs episodes.

        :param loss: the array with the losses.
        :param title: the plot's title.
        :param filename_suffix: the saved plot's filename suffix.
        """
        fig, ax = self._plot_score_vs_episodes(loss, title, 'Loss')
        self._plot_and_save(fig, self.plots_name_prefix + filename_suffix)
