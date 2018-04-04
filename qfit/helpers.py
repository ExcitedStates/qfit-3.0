import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class DJoiner(object):
    
    """Join filenames with a set directory."""

    def __init__(self, directory):
        self.directory = directory

    def __call__(self, fname):
        return os.path.abspath(os.path.join(self.directory, fname))
