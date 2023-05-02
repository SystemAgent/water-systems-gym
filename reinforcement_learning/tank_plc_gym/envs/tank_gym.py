import os
import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize
from gym.spaces import Dict, Discrete, Box, Tuple
from gym import error, core
import wntr

from reinforcement_learning.tank_plc_gym.envs.tank_model import WaterTank
from reinforcement_learning.tank_plc_gym.utils import observation_validation
from reinforcement_learning.config import PROJECT_PATH


class TankGym(core.Env):
    """Gym-like environment for water distribution systems.

    Args:
    Returns:
        A gym environment
    """

    def __init__(self,
                 gym_name='basic_tank',
                 patterns=None,
                 dt=10,
                 valve_cmd=0.5,
                 tank_level=50,
                 qout=0,
                 episode_len=600,
                 random_steps=0,
                 seed=None,
                 model_name='ppo_disc'):

        self.seed_number = seed
        if self.seed_number:
            np.random.seed(self.seed_number)
        else:
            np.random.seed()

        self.dt = dt
        self.valve_cmd = valve_cmd
        self.tank_level = tank_level
        self.qout = qout
        self.steps = 0
        self.model_name = model_name

        # number of random steps before training
        self.number_of_random_steps = random_steps

        # TODO real demand pattern
        if patterns is not None:
            self.patterns = patterns
        else:
            bull = [i for i in range(20, 100)]
            bear = [i for i in range(100, 20, -1)]
            self.pattern = (bull + bear)
            self.patterns = np.array([self.pattern])
        pattern = list(self.patterns[np.random.randint(
            self.patterns.shape[0], size=1), :][0])  # get random pattern for the next episode
        self.tank = WaterTank(self.dt, self.valve_cmd,
                              self.qout, pattern, self.tank_level)

        # Reward control
        self.episode_length = episode_len
        self.rewScale = [5, 8, 3]
        self.base_reward = +1
        self.bump_penalty = -1
        self.distance_range = .5
        self.wrong_move_penalty = -1
        self.laziness_reward = 0.5

        # Reward tweaking

        self.max_reward = +1
        self.min_reward = -1

        # self.optimized_speeds.fill(np.nan)
        observation = self.reset(training=False)
        self.optimized_value = np.nan
        self.previous_distance = np.nan

        basic_Box_observation_space = Box(
            low=0, high=+100, shape=(3,), dtype=np.float32)
        basic_Box_action_space = Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.config_dict = {'ppo_disc': {'action_space': Discrete(7),
                                         'observation_space': basic_Box_observation_space,
                                         'step_function': self.step_action},
                            'ppo_cont': {'action_space': basic_Box_action_space,
                                         'observation_space': basic_Box_observation_space,
                                         'step_function': self.continuous_action},
                            'appo': {'action_space': basic_Box_action_space,
                                     'observation_space': basic_Box_observation_space,
                                     'step_function': self.continuous_action},
                            'ddpg': {'action_space': basic_Box_action_space,
                                     'observation_space': basic_Box_observation_space,
                                     'step_function': self.continuous_action},
                            'dqn': {'action_space': Discrete(7),
                                    'observation_space': basic_Box_observation_space,
                                    'step_function': self.step_action}}
        # Init of observation, steps, done

        self.action_space = self.config_dict[self.model_name]['action_space']
        self.observation_space = self.config_dict[self.model_name]['observation_space']

    def step(self, action, debug=False):
        """ Reward computed from the Euclidean distance between the speed of the pumps
                   and the optimized speeds. """

        self.steps += 1
        self.done = (self.steps == self.episode_length)

        if debug:
            import ipdb
            ipdb.set_trace()
        if action not in self.action_space:
            raise error.Error(
                'Action out of the environment action space provided.'
                'Step function requires an action from the env.action_space.')

        self.config_dict[self.model_name]['step_function'](action)
        laziness_increment = self.increment_laziness(action)
        self.tank.calculateAll()
        reward = self.get_state_value() + laziness_increment

        observation = self.get_observation()
        observation_validation(observation)
        return observation, reward, self.done, {}

    def disc_lazy_increment(self, action):
        laziness_increment = 0
        if not isinstance(self.action_space, Discrete):
            raise error.Error(
                'Action space must be Discrete for usage of this function.')
        else:
            if action == 0:
                laziness_increment = 2.5
            else:
                laziness_increment = 0
        return laziness_increment

    def continuous_lazy_increment(self, action):
        if not isinstance(self.action_space, Box):
            raise error.Error(
                'Action space must be Box for usage of this function.')
        else:
            diff = np.absolute(action - self.tank.valve_cmd)
            if diff == 0:
                lazy_cont_increment = 5.5
            elif diff <= 0.3:
                lazy_cont_increment = 7.5
            else:
                lazy_cont_increment = 0
            return float(lazy_cont_increment)

    def continuous_differntial_increment(self, action, max_increment=2):
        diff = np.absolute(action[0] - self.tank.valve_cmd)
        if diff > 1:
            # TODO custom class for exception handling
            raise IndexError('action out of range')
        return max_increment - diff * max_increment

    def increment_laziness(self, action):
        laziness_increment = 0

        if isinstance(self.action_space, Box):
            laziness_increment = self.continuous_differntial_increment(action)
        if isinstance(self.action_space, Discrete):
            laziness_increment = self.disc_lazy_increment(action)
        return laziness_increment

    def increment_action(self, action):
        if action == 1:
            # increase
            self.tank.valve_cmd += 0.05
            if self.tank.valve_cmd >= 1:
                self.tank.valve_cmd = 1

        if action == 2:
            # decrease
            self.tank.valve_cmd -= 0.05
            if self.tank.valve_cmd <= 0:
                self.tank.valve_cmd = 0

        if action == 3:
            # increase
            self.tank.valve_cmd += 0.10
            if self.tank.valve_cmd >= 1:
                self.tank.valve_cmd = 1

        if action == 4:
            # decrease
            self.tank.valve_cmd -= 0.10
            if self.tank.valve_cmd <= 0:
                self.tank.valve_cmd = 0

        if action == 5:
            # increase
            self.tank.valve_cmd += 0.20
            if self.tank.valve_cmd >= 1:
                self.tank.valve_cmd = 1

        if action == 6:
            # decrease
            self.tank.valve_cmd -= 0.20
            if self.tank.valve_cmd <= 0:
                self.tank.valve_cmd = 0

        # if action == 'increase':
        #     # increase
        #     self.tank.valve_cmd += 0.5
        #     if self.tank.valve_cmd >= 1:
        #         self.tank.valve_cmd = 1
        # if action == 'decrease':
        #     # decrease
        #     self.tank.valve_cmd -= 0.5
        #     if self.tank.valve_cmd <= 0:
        #         self.tank.valve_cmd = 0
        # self.tank.calculateAll()
        # reward = self.get_state_value()
        # # if action[0] or action[1] == 0:
        # #     reward += self.laziness_reward
        # if action == ' hold':
        #     reward += self.laziness_reward

    def step_action(self, action):

        if action == 1:
            # step 1
            self.tank.valve_cmd = 0.0

        if action == 2:
            # step 2
            self.tank.valve_cmd = 0.20

        if action == 3:
            # step 3
            self.tank.valve_cmd = 0.40

        if action == 4:
            # step 4
            self.tank.valve_cmd = 0.60

        if action == 5:
            # step 5
            self.tank.valve_cmd = 0.80

        if action == 6:
            # step 6
            self.tank.valve_cmd = 1

    def continuous_action(self, action):
        if action[0] > 1:
            action[0] = 1
        if action[0] < 0:
            action[0] = 0
        self.tank.valve_cmd = np.round(action, 4)[0]

    def get_observation(self):
        # if self.steps >= 639:
        #     print(np.array([self.tank.tank_level,
        #                     self.tank.valve_cmd, self.tank.qout_next]))
        # return np.array([self.tank.tank_level, self.tank.valve_cmd, self.tank.qout_avg])
        return np.array([self.tank.tank_level, self.tank.valve_cmd, self.tank.qout_next])

    def get_state_value_initial(self):
        if 0 <= self.tank.tank_level < 15:
            reward = -100
        if 15 <= self.tank.tank_level < 18:
            reward = -10
        if 18 <= self.tank.tank_level < 25:
            reward = -1
        if 25 <= self.tank.tank_level < 75:
            reward = 10
        if 75 <= self.tank.tank_level < 82:
            reward = -1
        if 82 <= self.tank.tank_level < 85:
            reward = -10
        if 85 <= self.tank.tank_level <= 100:
            reward = -100

        # print(reward)
        return np.round(reward, 4)

    def get_state_value(self):
        '''
        The tank level is a percentage of it's capacity
        Reward is based on the tank level as:
            - if level is between 0 and 40 or 60 and 100 the reward is negative
            - if level is between 40 and 50 the reward gradually increases as it's get's closer to 50
            - if level is between 50 and 60 the reward gradually decreases as it's get's further from 50
        '''
        if 0 <= self.tank.tank_level < 20:
            reward = -10 - ((20 - self.tank.tank_level) / (20 - 0)) * 90
        if 20 <= self.tank.tank_level < 25:
            reward = -2 - ((25 - self.tank.tank_level) / (25 - 20)) * 8
        if 25 <= self.tank.tank_level < 40:
            reward = 0 - ((40 - self.tank.tank_level) / (40 - 25)) * 2
        if 40 <= self.tank.tank_level < 50:
            # scale interval [40:50] to [0:2]
            reward = 8 + ((self.tank.tank_level - 40) / (50 - 40)) * 2
        if 50 <= self.tank.tank_level < 60:
            # scale interval [50:60] to [0:2]
            reward = 10 - ((self.tank.tank_level - 50) / (60 - 50)) * 2
        if 60 <= self.tank.tank_level < 75:
            reward = 0 - ((self.tank.tank_level - 60) / (75 - 60)) * 2
        if 75 <= self.tank.tank_level < 80:
            reward = -2 - ((self.tank.tank_level - 75) / (80 - 75)) * 8
        if 80 <= self.tank.tank_level <= 100:
            reward = -10 - ((self.tank.tank_level - 80) / (100 - 80)) * 90

        return np.round(reward, 4)

    def reset(self, training=True, testing=False):
        pattern = list(self.patterns[np.random.randint(
            self.patterns.shape[0], size=1), :][0])  # get random pattern for the next episode

        if testing:  # get specific pattern for the next episode
            pattern = list(self.patterns[1])

        self.tank = WaterTank(self.dt, self.valve_cmd,
                              self.qout, pattern, self.tank_level)
        self.random_steps()
        observation = self.get_observation()
        observation_validation(observation)
        self.done = False
        self.steps = 0
        return observation

    def random_steps(self):
        for i in range(self.number_of_random_steps):
            self.tank.valve_cmd = np.random.uniform(
                low=0.0, high=1.0, size=1)[0]
            self.tank.calculateAll()

    def set_pattern(self, pattern):
        self.tank.qout_pattern = np.array(pattern)

    def seed(self, seed=None):
        """Collecting seeds."""
        return [seed]
