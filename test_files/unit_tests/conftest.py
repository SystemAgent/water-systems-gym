import os
import yaml

import pytest
import pandas as pd

from reinforcement_learning.config import TESTING, PROJECT_PATH, ROOT_PATH
from reinforcement_learning.tank_plc_gym.envs.tank_gym import TankGym


if not TESTING:
    raise ValueError("Tests should be executed in testing environment")


def create_environment(test_file_name):
    """Create environment for testing"""
    path_to_params = os.path.join(
        ROOT_PATH, 'test_files', 'unit_tests', 'config_test_files', test_file_name)

    with open(path_to_params, 'r') as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    scene_file_name = hparams['evaluation']['scenes'] + '.csv'
    scene_path = os.path.join(
        PROJECT_PATH, 'tank_plc_gym', 'data', scene_file_name)
    pattern_scenes = pd.read_csv(scene_path, index_col=0)

    env = TankGym(
        gym_name=hparams['env']['gym_name'],
        patterns=pattern_scenes.to_numpy(),
        dt=hparams['env']['dt'],
        valve_cmd=hparams['env']['valve_cmd'],
        tank_level=hparams['env']['tank_level'],
        qout=hparams['env']['qout'],
        episode_len=hparams['env']['episode_len'],
        model_name=hparams['model']['type']
    )
    return env, hparams


@pytest.fixture
def tank_gym_env_disc():
    """Create environment for testing with Discrete Action space."""
    return create_environment(test_file_name='test_basicTank_Rllib_DQN.yaml')


@pytest.fixture
def tank_gym_env_cont():
    return create_environment(test_file_name='test_basicTank_Rllib_cont_PPO.yaml')
