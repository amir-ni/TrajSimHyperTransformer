import yaml
import argparse
import torch
import random
import numpy as np

from logger import get_logger
from train import run_train
from test import run_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None, help="train or test")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    mode = args.mode if args.mode else config['general']['mode']
    seed = config['general']['seed']
    set_seed(seed)

    logger = get_logger(name='HyperTransformer', log_dir='logs/', log_level=20)
    logger.info(f"Running with config: {args.config}, mode={mode}")

    if mode=='train':
        run_train(config, logger)
    elif mode=='test':
        run_test(config, logger)
    else:
        logger.error(f"Unknown mode: {mode}")

if __name__=="__main__":
    main()
