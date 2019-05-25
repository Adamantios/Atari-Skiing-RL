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


def print_progressbar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 0,
                      length: int = 50, fill: str = '='):
    """
    Call in a loop to create terminal progress bar.
    @params:
        iteration    - Required  : current iteration (Int)
        total        - Required  : total iterations (Int)
        prefix       - Optional  : prefix string (Str)
        suffix       - Optional  : suffix string (Str)
        decimals     - Optional  : positive number of decimals in percent complete (Int)
        length       - Optional  : character length of bar (Int)
        fill         - Optional  : bar fill character (Str)
        clean_update - Optional  : if the update should leave a new line after the progressbar (bool)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    finish = '>' if iteration != 0 else ''
    bar = fill * filled_length + finish + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')

    # Print New Line on Complete.
    if iteration == total:
        print()
