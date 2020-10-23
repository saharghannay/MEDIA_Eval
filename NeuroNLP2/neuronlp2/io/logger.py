_author__ = 'max'

import logging
import sys
from logging.handlers import RotatingFileHandler
from logging import handlers

def get_logger(name, handler2, level=logging.INFO, handler1=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler1)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    fh = handlers.RotatingFileHandler(handler2, maxBytes=(1048576*5), backupCount=7)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
