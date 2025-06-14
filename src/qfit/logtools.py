import sys
import os
import time
import logging
import logging.handlers
import multiprocessing as mp
import numpy as np
import threading
#import pkg_resources  # deprecated
from importlib.metadata import version, PackageNotFoundError

from . import LOGGER as MODULELOGGER


def setup_logging(options, filename="qfit.log"):
    """Attaches logging handlers to module logger with appropriate loglevels.

    Args:
        options (_BaseQFitOptions): A QFitOptions object.
    """
    # Determine logger levels for handlers
    if options.debug:
        file_log_level = logging.DEBUG
        console_log_level = logging.DEBUG
    elif options.verbose:
        file_log_level = logging.INFO
        console_log_level = logging.INFO
    else:
        file_log_level = logging.INFO
        console_log_level = logging.WARNING

    # Create formatter
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] " "%(processName)-10s %(name)s : %(message)s"
    )

    # Create & attach console loghandler
    console_loghandler = logging.StreamHandler(stream=sys.stdout)
    console_loghandler.setLevel(console_log_level)
    console_loghandler.setFormatter(log_formatter)
    MODULELOGGER.addHandler(console_loghandler)

    # Create & attach file loghandler
    logging_fname = os.path.join(options.directory, filename)
    file_loghandler = logging.FileHandler(filename=logging_fname, mode="a")
    file_loghandler.setLevel(file_log_level)
    file_loghandler.setFormatter(log_formatter)
    MODULELOGGER.addHandler(file_loghandler)

    # Drop level of the modulelogger, so things get passed to handlers?
    MODULELOGGER.setLevel(min(file_log_level, console_log_level))


def log_run_info(options, logger):
    """Prints run info to a logger object.

    Args:
        options (_BaseQFitOptions): A QFitOptions object for this run.
        logger (logging.Logger): The logger that will log the messages.
    """
    
    #version = pkg_resources.require("qfit")[0].version  # deprecated
    try:
        qfit_version = version("qfit")
    except PackageNotFoundError:
        qfit_version = "unknown"

    cmd = " ".join(sys.argv)
    logger.info(f"===== qFit version: {qfit_version} =====")
    logger.info(time.strftime("%c %Z"))
    logger.info(f"{cmd}")
    logger.info(f"===== qFit parameters: =====")
    for key in vars(options).keys():
        logger.info(f"{key}: {getattr(options, key)}")
    logger.info(f"numpy float resolution: {np.finfo(float).resolution:.1e}")
    logger.info(f"============================\n")


def poolworker_setup_logging(logqueue):
    """Attaches a QueueHandler to the RootLogger of a subprocess.

    The MainProcess of qFit should handle all the logging. In particular, we
    attach a ConsoleHandler and FileHandler to the `qfit` module logger (see
    `setup_logging` above).

    However, subprocesses (i.e. PoolWorkers) do not have any access to the
    logger objects in MainProcess, and so their logs are never handled and
    aren't presented to the user.

    Each PoolWorker process has an independent RootLogger.
    This function attaches a QueueHandler to a PoolWorker's RootLogger, which
    handles all log records within the PoolWorker by putting them into
    the multiprocessing.Queue `logqueue`.

    Ideally, there'll be a QueueListener on MainProcess that will shuffle this
    Queue into the MainProcess logging system.

    Args:
        logqueue (multiprocessing.Queue): A queue to log messages to.
    """
    # Only if we are in a PoolWorker
    if mp.current_process().name != "MainProcess":
        # Get the RootLogger of this Process
        process_ROOTLOGGER = logging.getLogger()
        process_ROOTLOGGER.level = logging.NOTSET
        # If we haven't already set it up, attach a QueueHandler
        if len(process_ROOTLOGGER.handlers) == 0:
            queue_handler = logging.handlers.QueueHandler(logqueue)
            process_ROOTLOGGER.addHandler(queue_handler)


class QueueListener(threading.Thread):
    """A Thread that handles LogRecords from a Queue.

    Independent subprocesses (e.g. PoolWorkers) will place LogRecords onto a
    Queue. This QueueListener will process these LogRecords, and handle them
    in MainProcess. This way, we provide a way for subprocesses to access the
    logging Handlers in MainProcess.

    Run QueueListener.stop() to finish the process before QueueListener.join().

    Args:
        queue (multiprocessing.Queue): The queue that should be listened to.
            Contains log messages from other Processes which are to be
            'handled' here in MainProcess.
    """

    def __init__(self, queue, **kwargs):
        super().__init__(**kwargs)
        self._queue = queue
        self._terminator = threading.Event()

    def stop(self):
        self._terminator.set()

    def run(self):
        """Processes messages in queue via logging."""
        while not self._terminator.is_set():
            while not self._queue.empty():
                record = self._queue.get()
                logger = logging.getLogger(record.name)
                logger.handle(record)
            time.sleep(0.5)
