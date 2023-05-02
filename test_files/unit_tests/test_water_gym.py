import os
import gym
import numpy as np
import pytest
import yaml
from gym import error
from reinforcement_learning.pump_gym.envs.water_gym import WaterGym
from reinforcement_learning.config import PROJECT_PATH


class TestWaterGymEnvironment:
    test_class = WaterGym
    key = ''

    @pytest.fixture
    def water_gym_env(self):
        """Create environment for testing"""
        path_to_params = os.path.join(
            PROJECT_PATH, 'pump_gym', 'experiments', 'hyperparameters', 'anytownMaster.yaml')
        with open(path_to_params, 'r') as fin:
            hparams = yaml.load(fin, Loader=yaml.Loader)
        env = self.test_class(
            wn_name=hparams['env']['water_network'] + '_master',
            speed_increment=hparams['env']['speed_increment'],
            episode_len=hparams['env']['episode_length'],
            pump_groups=hparams['env']['pump_groups'],
            total_demand_low=hparams['env']['total_demand_low'],
            total_demand_high=hparams['env']['total_demand_high'],
            seed=42
        )
        return env

    def test_make_env(self):
        if self.key != '':
            env = gym.make(self.key)
            assert type(env) == self.test_class

    def test_initialization_types(self, water_gym_env):
        env = water_gym_env

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise error.Error(
                'WaterGym environment requires an observation space of type gym.spaces.Box')
        assert type(env.observation_space) == gym.spaces.Box

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise error.Error(
                'WaterGym environment requires an action space of type gym.spaces.Discrete')
        assert type(env.action_space) == gym.spaces.Discrete

    def test_step_no_action(self, water_gym_env):
        action = []
        with pytest.raises(Exception):
            assert water_gym_env.step(action), 'No action provided.' \
                                               ' Step function requires an action from the env.action_space.'

    def test_step_wrong_action(self, water_gym_env):
        action = 10
        with pytest.raises(Exception):
            assert water_gym_env.step(action), 'Action out of the environment action space provided.' \
                                               ' Step function requires an action from the env.action_space.'

    def test_reset(self, water_gym_env):
        assert type(water_gym_env.reset()) == np.ndarray

    def test_observation(self, water_gym_env):
        with pytest.raises(Exception):
            assert isinstance(water_gym_env.reset(),
                              water_gym_env.observation_space)

    def test_pump_efficiencies_speeds(self, water_gym_env):
        with pytest.raises(Exception):
            assert type(water_gym_env.pump_efficiencies) == np.ndarray
            assert type(water_gym_env.pumpEffs) == np.ndarray
            assert type(water_gym_env.pump_speeds) == np.ndarray
            assert water_gym_env.pump_efficiencies.shape == len(
                water_gym_env.pump_groups)
            assert water_gym_env.pumpEffs.shape == len(
                water_gym_env.pump_groups)
            assert water_gym_env.pump_speeds.shape == len(
                water_gym_env.pump_groups)

    def test_valid_optimized_speeds(self, water_gym_env):
        with pytest.raises(Exception):
            assert type(water_gym_env.valid_speeds) == np.float32
            assert type(water_gym_env.optimized_speeds) == np.ndarray

# TODO test_network_no_curves(self); test_randomize_demands(); test_get_state_value()
#  test_calculate_pump_efficiencies; test_update_pump_speeds(); test_get_pump_speeds(); test_get_pump_efficiencies()
