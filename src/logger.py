import logging
import os
import sys

def get_logger(name='HyperTransformer', log_dir='logs/', log_level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        # console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(ch_format)
        logger.addHandler(ch)

        # file
        fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        fh.setLevel(log_level)
        fh_format = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s')
        fh.setFormatter(fh_format)
        logger.addHandler(fh)
    return logger
