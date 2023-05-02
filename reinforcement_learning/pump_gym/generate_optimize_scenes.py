import argparse
import os
import random
import array
import multiprocessing

import yaml
import operator
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead

from reinforcement_learning.config import PROJECT_PATH
from reinforcement_learning.pump_gym.envs.water_gym import WaterGym


parser = argparse.ArgumentParser()
parser.add_argument('--params', default='anytownMaster',
                    type=str, help="Name of the YAML file.")
parser.add_argument('--nscenes', default=100, type=int,
                    help="Number of the scenes to generate.")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed for the optimization methods.")
parser.add_argument('--scenes', default=None, type=str,
                    help="Name of the generated scenes file.")
parser.add_argument('--result', default=None, type=str,
                    help="Name of the generated result file.")
parser.add_argument('--nproc', default=1, type=int,
                    help="Number of processes to raise.")
args = parser.parse_args()

path_to_params = os.path.join(
    PROJECT_PATH, 'pump_gym', 'experiments', 'hyperparameters', args.params+'.yaml')
with open(path_to_params, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)

reset_orig_demands = hparams['env']['reset_original_demands']
wn_name = hparams['env']['water_network']+'_master'

if args.scenes:
    scene_file_name = args.scenes + '.csv'
else:
    scene_file_name = hparams['evaluation']['scenes'] + '.csv'

if args.result:
    result_file_name = args.result + '.csv'
else:
    result_file_name = hparams['evaluation']['result'] + '.csv'

n_scenes = args.nscenes
seed = args.seed
n_proc = args.nproc

if seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
else:
    random.seed()
    np.random.seed()

if n_scenes > 10:
    verbosity = n_scenes // 10
else:
    verbosity = 1

env = WaterGym(
    wn_name=hparams['env']['water_network']+'_master',
    speed_increment=hparams['env']['speed_increment'],
    episode_len=hparams['env']['episode_length'],
    pump_groups=hparams['env']['pump_groups'],
    total_demand_low=hparams['env']['total_demand_low'],
    total_demand_high=hparams['env']['total_demand_high'],
    reset_original_demands=hparams['env']['reset_original_demands'],
    reset_original_pump_speeds=hparams['env']['reset_original_pump_speeds'],
    seed=args.seed
)


def reward_to_scipy(pump_speeds):
    # print('optimizing speed')
    # print(pump_speeds)
    """Only minimization allowed."""
    return -env.get_state_value_to_opti(pump_speeds)


def generate_scenes(reset_orig_demands, n_scenes):
    junction_ids = list(env.wn.junction_name_list)
    demand_df = pd.DataFrame(
        np.empty(shape=(n_scenes, len(junction_ids))),
        columns=junction_ids)
    if reset_orig_demands:
        for i in range(n_scenes):
            demand_df.loc[i] = env.build_demand_dict()
    else:
        for i in range(n_scenes):
            env.randomize_demands()
            demand_df.loc[i] = env.build_demand_dict()

    return demand_df


class Nelder_mead_method():
    def __init__(self):
        self.options = {'maxfev': 1000, 'xatol': .005, 'fatol': .01}

    def maximize(self, scene_id):
        if seed:
            random.seed(args.seed)
            init_guess = []
            for i in range(env.dimensions):
                init_guess.append(random.uniform(
                    env.speed_limit_low, env.speed_limit_high))
        else:
            random.seed()
            init_guess = []
            for i in range(env.dimensions):
                init_guess.append(random.uniform(
                    env.speed_limit_low, env.speed_limit_high))

        env.set_demands(scene_df.loc[scene_id])
        # init_guess = [1.2, 0.83369588, 0.87215528, 1.04711067, 1.07251507]
        options = {'maxfev': 1000, 'xatol': .005, 'fatol': .01}
        result = neldermead(
            reward_to_scipy,
            init_guess,
            method='Nelder-Mead',
            options=options)

        result_df = pd.DataFrame(
            np.empty(shape=(1, len(df_header))),
            columns=df_header)
        result_df['index'] = scene_id
        result_df['reward'] = -result.fun
        result_df['evals'] = result.nit

        for i in range(env.dimensions):
            result_df['speed_of_group'+str(i)] = result.x[i]
        return result_df


def optimize_scenes(scene_df, method=None):
    # pool        = multiprocessing.Pool(n_proc)
    # result_df   = pool.map(method, range(len(scene_df)))
    result_df = map(method, range(len(scene_df)))
    result_df = pd.concat(result_df)
    result_df.set_index('index', inplace=True)
    result_df.rename_axis(None, inplace=True)
    return result_df


scene_df = generate_scenes(reset_orig_demands, n_scenes)
scene_df.to_csv(os.path.join(
    PROJECT_PATH, 'pump_gym', 'data', scene_file_name))

df_header = ['index', 'reward', 'evals']
for i in range(env.dimensions):
    df_header.append('speed_of_group'+str(i))

nm = Nelder_mead_method()
subdf_nm = optimize_scenes(scene_df, nm.maximize)

subdf_nm.to_csv(os.path.join(
    PROJECT_PATH, 'pump_gym', 'data', result_file_name))
