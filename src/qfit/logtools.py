'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

import sys
import os
import pkg_resources
import time
import logging
import logging.handlers
import multiprocessing as mp
import threading

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
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(processName)-10s %(name)s : %(message)s')

    # Create & attach console loghandler
    console_loghandler = logging.StreamHandler(stream=sys.stdout)
    console_loghandler.setLevel(console_log_level)
    console_loghandler.setFormatter(log_formatter)
    MODULELOGGER.addHandler(console_loghandler)

    # Create & attach file loghandler
    logging_fname = os.path.join(options.directory, filename)
    file_loghandler = logging.FileHandler(filename=logging_fname, mode='a')
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
    version = pkg_resources.require("qfit")[0].version
    cmd = " ".join(sys.argv)
    logger.info(f"===== qFit version: {version} =====")
    logger.info(time.strftime("%c %Z"))
    logger.info(f"{cmd}")
    logger.info(f"===== qFit parameters: =====")
    for key in vars(options).keys():
        logger.info(f"{key}: {getattr(options, key)}")
    logger.info(f"============================\n")
