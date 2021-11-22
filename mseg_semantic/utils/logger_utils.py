"""Utilities for setting up a Python logger."""

import logging


def get_logger() -> logging.Logger:
    """ Set up a logger to print out log messages in the following format:
    {TIME} {LOGGING LEVEL} {FILENAME GENERATING THE LOG MESSAGE} line {LINE NUMBER} {LOG MESSAGE}
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
