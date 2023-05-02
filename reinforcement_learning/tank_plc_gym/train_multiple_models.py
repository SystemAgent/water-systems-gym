import os
import itertools

import pandas as pd
import numpy as np
import argparse
import yaml
import pprint
from datetime import datetime

from reinforcement_learning.tank_plc_gym.rllib_agents import train
from reinforcement_learning.config import PROJECT_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--params', default='basicTankRllibMultipleModels',
                    help="Name of the YAML file.")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed for the optimization methods.")
parser.add_argument('--workers', default=1, type=int,
                    help="Number of processes to raise.")

args = parser.parse_args()

root_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym')
experiment_path = os.path.join(root_path, 'experiments')
data_path = os.path.join(root_path, 'data')
parameters_path = os.path.join(
    experiment_path, 'hyperparameters', args.params+'.yaml')

grid_search_parameter_names = [('model', 'layers'),
                               ('training', 'initial_learning_rate'),
                               ('training', 'timesteps_per_iteration'),
                               ('training', 'number_of_iterations'),
                               ('env', 'dt'),
                               ('env', 'tank_level')]

if __name__ == '__main__':
    with open(parameters_path, 'r') as fin:
        hparams_multiple = yaml.load(fin, Loader=yaml.Loader)

    grid_search_parameters = [
        hparams_multiple[parameter[0]][parameter[1]] for parameter in grid_search_parameter_names]

    pp = pprint.PrettyPrinter(indent=4)

    for index, grid_parameteres in enumerate(itertools.product(*grid_search_parameters)):
        for parameter in zip(grid_search_parameter_names, grid_parameteres):
            hparams_multiple[parameter[0][0]][parameter[0][1]] = parameter[1]

        print('train new model')
        pp.pprint(grid_parameteres)

        train(hparams_multiple, experiment_name=str(
            index))
