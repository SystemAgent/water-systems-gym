import os

import pandas as pd
from datetime import datetime
import numpy as np
import ray
import shutil
import argparse
import yaml
import pprint
import gym
from ray import tune
from ray.tune.registry import register_env
from datetime import datetime
from ray.tune.callback import Callback
from ray.tune.integration.mlflow import MLflowLoggerCallback
import mlflow

from reinforcement_learning.tank_plc_gym.utils import main_visualization, scene_evaluation, create_agent, CheckpointCallback, function_timer
from reinforcement_learning.tank_plc_gym.envs.tank_gym import TankGym
from reinforcement_learning.services.mlflow.utils import MlflowClient
from reinforcement_learning.config import PROJECT_PATH, MLFLOW_USER, MLFLOW_PASSWORD, MLFLOW_TRACKING_URI

os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD


parser = argparse.ArgumentParser()
parser.add_argument('--params', default='basicTank_Rllib_DQN',
                    help="Name of the YAML file.")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed for the optimization methods.")
parser.add_argument('--tstsplit', default=20, type=int,
                    help="Ratio of scenes moved from vld to tst scene in percentage.")

args = parser.parse_args()

root_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym')
experiment_path = os.path.join(root_path, 'experiments')
data_path = os.path.join(root_path, 'data')
parameters_path = os.path.join(
    experiment_path, 'hyperparameters', args.params+'.yaml')


@function_timer
def train(hparams, experiment_name=None):
    """Trains an Rllib model with tune.

    :param hparams: [yaml file containing the parameters for the particular model]
    :type hparams: [type]
    :param experiment_name: [name of the experiment in MLflow], defaults to None
    :type experiment_name: [type], optional
    :return: [ ExperimentAnalysis: Object for experiment analysis.]
    :rtype: [type]
    """
    chkpt_root = os.path.join(experiment_path, 'history', experiment_name)
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    if experiment_name:
        data_path = os.path.join(root_path, 'data', experiment_name)
    else:
        data_path = os.path.join(root_path, 'data')

    # clear previous model data
    shutil.rmtree(data_path, ignore_errors=True, onerror=None)

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    scene_file_name = hparams['evaluation']['scenes'] + '.csv'
    scene_path = os.path.join(
        PROJECT_PATH, 'tank_plc_gym', 'data', scene_file_name)

    ray.init(ignore_reinit_error=True)

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

    _, agent, config = create_agent(select_env=select_env, hparams=hparams,
                                    agent_type=hparams['model']['type'])

    config['env'] = select_env
    # config['mlflow'] = {'experiment_name': 'New Experiment'}
    n_iter = hparams['training']['number_of_iterations']
    env = TankGym(**env_parameters)
    a = datetime.now()
    analysis = tune.run(agent,
                        config=config,
                        stop={
                            "training_iteration": n_iter},
                        # verbose=0,
                        checkpoint_freq=n_iter//5,
                        local_dir=chkpt_root,
                        checkpoint_at_end=True,
                        callbacks=[
                            CheckpointCallback(
                                data_path=data_path, env=env, pattern_scenes=pattern_scenes, select_env=select_env, hparams=hparams)],
                        # MLflowLoggerCallback(experiment_name="Mark", tracking_uri=MLFLOW_TRACKING_URI)],
                        name="DQN_MARK")
    ray.shutdown()
    b = datetime.now()
    print(f'TOTAL TIME for {n_iter} iterations: {b - a}')
    return analysis


if __name__ == '__main__':
    with open(parameters_path, 'r') as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)
    train(hparams=hparams, experiment_name='test1')
