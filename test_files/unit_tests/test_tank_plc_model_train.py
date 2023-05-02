import os
import json
import yaml
import tempfile

import gym
import numpy as np
import pytest
from gym import error
import ray
from ray.tune.registry import register_env

from reinforcement_learning.config import ROOT_PATH
from reinforcement_learning.tank_plc_gym.utils import create_agent
from reinforcement_learning.tank_plc_gym.rllib_train import train


def test_dqn_disc_one_iteration(tank_gym_env_disc):
    env, hparams = tank_gym_env_disc

    register_env('tank_env_disc', lambda config: env)

    agent, _, _ = create_agent(
        select_env='tank_env_disc', hparams=hparams, agent_type=hparams['model']['type'])

    for n in range(1, 1):
        result = agent.train()

    with tempfile.TemporaryDirectory(prefix='dqn_rlib') as tmpdir:
        chkpt_file = agent.save(tmpdir)

        assert os.path.exists(os.path.join(
            tmpdir, 'checkpoint_000000')) == True


def test_ppo_cont_one_iteration(tank_gym_env_cont):
    env, hparams = tank_gym_env_cont

    register_env('tank_env_cont', lambda config: env)

    agent, _, _ = create_agent(
        select_env='tank_env_cont', hparams=hparams, agent_type=hparams['model']['type'])

    for n in range(1, 1):
        result = agent.train()

    with tempfile.TemporaryDirectory(prefix='ppo_rlib') as tmpdir:
        chkpt_file = agent.save(tmpdir)

        assert os.path.exists(os.path.join(
            tmpdir, 'checkpoint_000000')) == True


def test_rllib_train(mocker):
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'basicTank_Rllib_DQN.yaml'), 'r') as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with open(os.path.join(os.path.dirname(__file__),
                           'model_train_data', 'iteration_result.json')) as f:
        result = json.load(f)
    mocker.patch('reinforcement_learning.tank_plc_gym.utils.DQNTrainer.train',
                 return_value=result)

    mocker.patch(
        'reinforcement_learning.tank_plc_gym.rllib_train.log_rllib_metrics_mlflow')

    mocker.patch(
        'reinforcement_learning.tank_plc_gym.rllib_train.data_path', os.path.join(os.path.dirname(__file__), 'data'))
    train(hparams=hparams, experiment_name='test_result',
          checkpoint_experiment='test_experiment', checkpoint_number=2, checkpoint_frequency=1)

    assert os.path.exists(os.path.join(
        os.path.join(os.path.dirname(__file__), 'data', 'test_result', 'checkpoint_rllib0'))) == True
