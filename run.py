import os
import argparse
import yaml

import torch
import numpy as np
import random

import logging
import wandb

from experiment import Experiment

# If you do not want Wandb, uncomment the following line
#os.environ["WANDB_MODE"] = "disable"

def parse_arguments():
    """
    Parse the arguments from the command line
    """
    parser = argparse.ArgumentParser(description='Training a 3D density field for TEM')
    parser.add_argument('config_file', type=str, help='Path to the yaml config file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def main():
    ## Reproducability
    torch.manual_seed(368)
    np.random.seed(368)
    random.seed(368)

    ## Default torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    ## Parsing arguments
    args = parse_arguments()
    if args.debug:
        #os.environ['WANDB_MODE'] = 'disabled'
        args.name += '_DEBUG'
        args.n_iter = 5000
        args.log_every_train = 10
        args.log_every_val = 100
        torch.autograd.set_detect_anomaly(True)

    ## Setting up logging and wandb
    logpath = os.path.join(args.logdir, args.name)
    os.makedirs(logpath, exist_ok=True)
    config_file = os.path.join(logpath, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    log_file = os.path.join(logpath, 'logs.log')
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='w', # Allows to overwrite and not append to an old log
                        level=logging.INFO)
    logging.info('Setting up wandb')
    wandb.init(project=args.project, name=args.name, config=args)

    ## Launching experiment
    logging.info('Launching experiment')
    exp = Experiment(args)
    exp.run()
    exp.save_checkpoint(logpath)

    logging.info('Done')

if __name__ == "__main__":
    main()