from argparse import ArgumentParser
from os import makedirs, path
import numpy as np


def rgb2gray(rgb) -> np.ndarray:
    """
    Converts an rgb image array to a grey image array.

    :param rgb: the rgb image array.
    :return: the converted array.
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def downsample(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Downsamples an image array, by a scale factor.

    :param img: the image to downsample.
    :param scale: the downsampling scale factor.
    :return: the downsampled image.
    """
    if scale < 2:
        return img

    return img[::scale, ::scale]


def atari_preprocess(frame_array: np.ndarray, downsample_scale: int = 2) -> np.ndarray:
    """
    Prepossesses the given atari frame array.

    :param frame_array: the atari frame array.
    :param downsample_scale: a scale to downsample the given array with.
    :return: the preprocessed frame array.
    """
    # Converting into greyscale since colors don't matter.
    greyscale_frame = rgb2gray(frame_array)

    # Downsampling the image.
    resized_frame = downsample(greyscale_frame, downsample_scale)

    # Reshape for batches and return.
    return np.reshape([resized_frame],
                      (1, frame_array.shape[0] // downsample_scale, frame_array.shape[1] // downsample_scale, 1))


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = path.dirname(filepath)

    # Create directory if it does not exist
    if not path.exists(directory) and not directory == '':
        makedirs(directory)
