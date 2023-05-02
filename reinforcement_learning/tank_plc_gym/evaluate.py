import os

import numpy as np
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.dqn as dqn
import shutil
import argparse
import yaml
import pprint
import gym
from ray.tune.registry import register_env

from reinforcement_learning.tank_plc_gym.utils import main_visualization, scene_evaluation, create_agent, find_checkpoint_path
from reinforcement_learning.tank_plc_gym.envs.tank_gym import TankGym
from reinforcement_learning.config import PROJECT_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--params', default='basicTank_Rllib_DQN',
                    help="Name of the YAML file.")
parser.add_argument('--name', default='evaluation',
                    help="Name output folder")
parser.add_argument('--experiment_name', default='test1',
                    help="Name of the experiment from which to load checkpoint ")
parser.add_argument('--checkpoint_number', default=4, type=int,
                    help="Number of the checkpoint to be restored.")
parser.add_argument('--scenes_file', default='basic_tank_evaluation_scenes',
                    help="Name of evaluation(test) scenes file")

args = parser.parse_args()

root_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym')
experiment_path = os.path.join(root_path, 'experiments')
history_path = os.path.join(experiment_path, 'history')
data_path = os.path.join(root_path, 'data')
parameters_path = os.path.join(
    experiment_path, 'hyperparameters', args.params+'.yaml')


def evaluate_agent(hparams):
    """ Function for evaluation of trained agents.

    :param hparams: [description]
    :type hparams: [type]
    """

    scene_file_name = hparams['evaluation']['scenes'] + '.csv'
    scene_path = os.path.join(
        PROJECT_PATH, 'tank_plc_gym', 'data', scene_file_name)

    checkpoint_path = find_checkpoint_path(
        data_path, args.experiment_name, args.checkpoint_number)

    pattern_scenes = pd.read_csv(scene_path, index_col=0)
    # register the custom environment
    select_env = "tankgym-v0"
    pattern_scenes = pd.read_csv(scene_path, index_col=0)
    env_parameters = {
        'gym_name': hparams['env']['gym_name'],
        'patterns': pattern_scenes.to_numpy(),
        'dt': hparams['env']['dt'],
        'valve_cmd': hparams['env']['valve_cmd'],
        'tank_level': hparams['env']['tank_level'],
        'qout': hparams['env']['qout'],
        'episode_len': hparams['env']['episode_len'],
        'model_name': hparams['model']['type']
    }

    register_env(select_env, lambda config: TankGym(**env_parameters))

    ray.init(ignore_reinit_error=True)
    agent, _, _ = create_agent(
        select_env=select_env, hparams=hparams, agent_type=hparams['model']['type'])
    agent.restore(checkpoint_path)
    print('AGENT LOADED')

    env = TankGym(**env_parameters)
    validation_path = os.path.join(
        data_path, str(args.name))

    # clear previous evaluation
    shutil.rmtree(validation_path, ignore_errors=True, onerror=None)

    if not os.path.exists(validation_path):
        os.mkdir(validation_path)

    result_dataframes = []

    validation_scenes_path = os.path.join(
        PROJECT_PATH, 'tank_plc_gym', 'data', args.scenes_file + '.csv')
    validation_scenes = pd.read_csv(validation_scenes_path, index_col=0)

    # apply the trained policy in a rollout
    scene_evaluation(pattern_scenes=validation_scenes, env=env,
                     agent=agent, validation_path=validation_path)


if __name__ == '__main__':
    with open(parameters_path, 'r') as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)
    evaluate_agent(hparams=hparams)
