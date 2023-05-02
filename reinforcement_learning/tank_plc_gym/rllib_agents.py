import os

import pandas as pd
import numpy as np
import ray
import shutil
import argparse
import yaml

from reinforcement_learning.tank_plc_gym.rllib_train import train
from reinforcement_learning.config import PROJECT_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--params', default='basicTank_Rllib_DQN',
                    help="Name of the YAML file.")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed for the optimization methods.")
parser.add_argument('--tstsplit', default=20, type=int,
                    help="Ratio of scenes moved from vld to tst scene in percentage.")
parser.add_argument('--experiment_name', default='test1',
                    help="Name of experiment from which to load checkpoint")
parser.add_argument('--checkpoint_experiment',
                    help="Name of experiment from which to load checkpoint")
parser.add_argument('--checkpoint_number', default=4, type=int,
                    help="Number of the checkpoint to be restored.")

args = parser.parse_args()

root_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym')
experiment_path = os.path.join(root_path, 'experiments')
data_path = os.path.join(root_path, 'data')
parameters_path = os.path.join(
    experiment_path, 'hyperparameters', args.params+'.yaml')


if __name__ == '__main__':
    with open(parameters_path, 'r') as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)
    train(hparams=hparams, experiment_name=args.experiment_name,
          checkpoint_experiment=args.checkpoint_experiment, checkpoint_number=args.checkpoint_number)
