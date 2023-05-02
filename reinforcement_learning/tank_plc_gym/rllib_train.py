import os

import pandas as pd
import numpy as np
import ray
import shutil
import argparse
import yaml
import pprint
import gym
from ray.tune.registry import register_env
from datetime import datetime
from ray.tune.logger import pretty_print


from reinforcement_learning.tank_plc_gym.utils import (main_visualization,
                                                       scene_evaluation,
                                                       create_agent,
                                                       log_rllib_metrics_mlflow,
                                                       function_timer,
                                                       find_checkpoint_path,
                                                       config_validation)
from reinforcement_learning.tank_plc_gym.envs.tank_gym import TankGym
from reinforcement_learning.config import PROJECT_PATH


root_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym')
data_path = os.path.join(root_path, 'data')


@function_timer
def train(hparams, experiment_name, checkpoint_experiment, checkpoint_number, checkpoint_frequency=5):
    """ Trains an Rllib model.

    :param hparams: [description]
    :type hparams: [type]
    :param experiment_name: [description]
    :type experiment_name: [type]
    :param checkpoint_experiment: [description]
    :type checkpoint_experiment: [type]
    :param checkpoint_number: [description]
    :type checkpoint_number: [type]
    :param checkpoint_frequency: [description], defaults to 5
    :type checkpoint_frequency: int, optional
    """
    config_validation(hparams)
    experiment_path = os.path.join(data_path, experiment_name)

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
        'random_steps': hparams['env']['random_steps'],
        'qout': hparams['env']['qout'],
        'episode_len': hparams['env']['episode_len'],
        'model_name': hparams['model']['type']
    }

    register_env(select_env, lambda config: TankGym(**env_parameters))

    agent, _, _ = create_agent(select_env=select_env, hparams=hparams,
                               agent_type=hparams['model']['type'])

    if checkpoint_experiment:
        checkpoint_path = find_checkpoint_path(
            data_path, checkpoint_experiment, checkpoint_number)
        agent.restore(checkpoint_path)
        print('AGENT LOADED')

    # clear previous model data
    shutil.rmtree(experiment_path, ignore_errors=True, onerror=None)

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    n_iter = hparams['training']['number_of_iterations']
    results = []
    iteration_results = []
    validation_number = 0
    validation_frequency = n_iter // checkpoint_frequency
    artifact_paths = []
    # train a policy with RLlib using PPO

    # no random steps for evaluation
    env_parameters['random_steps'] = 0

    for n in range(1, n_iter+1):
        result_dataframes = []
        a = datetime.now()
        result = agent.train()
        iteration_results.append(result)
        b = datetime.now()
        print(f'iteration number: {n}, took: {b - a}')
        if n % validation_frequency == 0:
            results.append(result)

            validation_path = os.path.join(
                experiment_path, 'checkpoint_' + 'rllib' + str(validation_number))
            if not os.path.exists(validation_path):
                os.mkdir(validation_path)

            chkpt_file = agent.save(validation_path)
            print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/'
                  f'{result["episode_reward_max"]:8.4f}. Checkpoint saved to {chkpt_file}')

            policy = agent.get_policy()
            env = TankGym(**env_parameters)

            # apply the trained policy in a rollout
            scene_evaluation(pattern_scenes=pattern_scenes, env=env,
                             agent=agent, validation_path=validation_path)

            artifact_paths.append(validation_path)
            validation_number += 1
    if hparams['model']['type'] == 'dqn':
        result = results
    if hparams['model']['type'] == 'ppo_disc' or 'ppo_cont':
        result = iteration_results

    df_result = pd.DataFrame(data=results)
    df_result.to_csv(experiment_path + '/results_file.csv')

    # Use only when you want to log experiment results in mlflow

    # log_rllib_metrics_mlflow(
    #     result=result, artifact_paths=artifact_paths,
    #     model_type=hparams['model']['type'],
    #     experiment_name='RLlib_' + hparams['model']['type'],
    #     run_name='rllib_checkpoints_' + str(validation_number - 1))
