import logging
from contextlib import contextmanager    

@contextmanager
def suppress_logging(level=logging.CRITICAL):
    """临时抑制 logging 输出"""
    previous_level = logging.root.manager.disable 
    logging.disable(level)     
    try:
        yield
    finally:
        logging.disable(previous_level)