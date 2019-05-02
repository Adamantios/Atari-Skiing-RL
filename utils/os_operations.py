from os import makedirs, path


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
