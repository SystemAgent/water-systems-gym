import os
import numpy as np

import scipy.stats as stats
from scipy.optimize import minimize
import gym.spaces
from gym import error
import wntr

from reinforcement_learning.pump_gym.optimization_algorithms import nm
from reinforcement_learning.config import PROJECT_PATH


class WaterGym:
    """Gym-like environment for water distribution systems."""

    def __init__(self,
                 gym_name='water_gym',
                 wn_name='anytown_master',
                 speed_increment=.05,
                 episode_len=10,
                 pump_groups=[['78', '79']],
                 total_demand_low=.3,  # Are there such equivalent boundaries of demand low/high?
                 total_demand_high=1.1,
                 reset_original_demands=False,
                 reset_original_pump_speeds=False,
                 seed=None):

        self.seed_number = seed
        if self.seed_number:
            np.random.seed(self.seed_number)
        else:
            np.random.seed()

        path_to_inp = os.path.join(
            PROJECT_PATH, 'pump_gym', 'water_networks', wn_name + '.inp')
        self.wn = wntr.network.WaterNetworkModel(path_to_inp)
        self.results = None

        self.demand_dict = self.build_demand_dict()
        self.pump_groups = pump_groups
        self.pump_speeds = np.ones(
            shape=(len(self.pump_groups)), dtype=np.float64)
        self.pump_efficiencies = np.empty(
            shape=(len(self.pump_groups)), dtype=np.float64)
        self.sum_of_demands = sum(
            [junction[1].base_demand for junction in self.wn.junctions()])
        self.demand_randomizer = self.build_truncnorm_randomizer(
            lo=.7, hi=1.3, mu=1.0, sigma=1.0)

        nominal_head_dictionary, nominal_efficiency_dictionary = self.get_curves_dicts()
        self.nominal_H_curve_polynomials = self.fit_polynomials(nominal_head_dictionary,
                                                                degree=2, encapsulated=True)
        self.nominal_E_curve_polynomials = self.fit_polynomials(nominal_efficiency_dictionary,
                                                                degree=4, encapsulated=True)

        # TODO abstract for more curves
        # Theoretical bounds of {head, efficiency}
        peak_heads = []
        for key in self.nominal_H_curve_polynomials.keys():
            max_q = np.max(nominal_head_dictionary[key][:, 0])
            opti_result = minimize(
                -self.nominal_H_curve_polynomials[key], x0=1, bounds=[(0, max_q)])
            peak_heads.append(
                self.nominal_H_curve_polynomials[key](opti_result.x[0]))
        peak_effs = []
        for key in self.nominal_H_curve_polynomials.keys():
            max_q = np.max(nominal_head_dictionary[key][:, 0])
            q_list = np.linspace(0, max_q, 10)
            head_poli = self.nominal_H_curve_polynomials[key]
            eff_poli = self.nominal_E_curve_polynomials[key]
            opti_result = minimize(-eff_poli, x0=1, bounds=[(0, max_q)])
            peak_effs.append(eff_poli(opti_result.x[0]))
        self.peak_total_efficiency = np.prod(peak_effs)

        # Reward control
        self.dimensions = len(self.pump_groups)
        self.episode_length = episode_len
        self.head_limit_low = 15  # find head low/high limit in WNTR sim
        self.head_limit_high = 120
        self.max_head = np.max(peak_heads)
        # eff, head, pump; What the hell is this for?
        self.rewScale = [5, 8, 3]
        self.base_reward = +1
        self.bump_penalty = -1
        self.distance_range = .5
        self.wrong_move_penalty = -1
        self.laziness_penalty = -1

        # Reward tweaking

        self.max_reward = +1
        self.min_reward = -1

        # Inner variables

        self.spec = None
        self.metadata = None
        self.total_demand_low = total_demand_low
        self.total_demand_high = total_demand_high
        self.speed_increment = speed_increment
        self.speed_limit_low = .7
        self.speed_limit_high = 1.2
        self.valid_speeds = np.arange(
            self.speed_limit_low,
            self.speed_limit_high + .001,
            self.speed_increment,
            dtype=np.float64)

        # TODO research meaning/difference of action and obervation space
        self.reset_original_pump_speeds = reset_original_pump_speeds
        self.reset_original_demands = reset_original_demands
        self.optimized_speeds = np.empty(
            shape=(len(self.pump_groups)), dtype=np.float64)

        # self.optimized_speeds.fill(np.nan)
        observation = self.reset(training=False)
        self.optimized_value = np.nan
        self.previous_distance = np.nan

        # Init of observation, steps, done
        self.action_space = gym.spaces.Discrete(2 * self.dimensions + 1)
        self.observation_space = gym.spaces.Box(low=-1,
                                                high=+1,
                                                shape=(
                                                    len(self.wn.junction_name_list) + len(self.pump_groups),),
                                                dtype=np.float64)

    def step(self, action, training=True):
        """ Reward computed from the Euclidean distance between the speed of the pumps
                   and the optimized speeds. """

        # Here , I think, we also have the only Control actions for the Agent
        self.steps += 1

        if action not in self.action_space:
            raise error.Error(
                'Action out of the environment action space provided.'
                ' Step function requires an action from the env.action_space.')

        group_id = action // 2  # what is the action
        command = action % 2  # what is the command
        if training:
            if group_id != self.dimensions:
                self.n_siesta = 0
                first_pump_in_group = self.wn.get_link(
                    self.pump_groups[group_id][0])
                if command == 0:
                    if first_pump_in_group.base_speed < self.speed_limit_high:
                        for pump_name in self.pump_groups[group_id]:
                            pump = self.wn.get_link(pump_name)
                            pump.base_speed += self.speed_increment
                        self.update_pump_speeds()
                        distance = np.linalg.norm(
                            self.optimized_speeds - self.pump_speeds)
                        if distance < self.previous_distance:
                            reward = distance * self.base_reward / self.distance_range / self.max_reward
                        else:
                            reward = self.wrong_move_penalty
                        self.previous_distance = distance
                    else:
                        self.n_bump += 1
                        reward = self.bump_penalty
                else:
                    if first_pump_in_group.base_speed > self.speed_limit_low:
                        for pump_name in self.pump_groups[group_id]:
                            pump = self.wn.get_link(pump_name)
                            pump.base_speed -= self.speed_increment
                        self.update_pump_speeds()
                        distance = np.linalg.norm(
                            self.optimized_speeds - self.pump_speeds)
                        if distance < self.previous_distance:
                            reward = distance * self.base_reward / self.distance_range / self.max_reward
                        else:
                            reward = self.wrong_move_penalty
                        self.previous_distance = distance
                    else:
                        self.n_bump += 1
                        reward = self.bump_penalty
            else:
                self.n_siesta += 1
                value = self.get_state_value()
                if self.n_siesta == 3:
                    self.done = True
                    if value / self.optimized_value > .98:
                        reward = 5 / self.max_reward
                    else:
                        reward = self.laziness_penalty
                else:
                    if value / self.optimized_value > .98:
                        reward = self.n_siesta * self.base_reward
                    else:
                        reward = self.laziness_penalty
            sim = wntr.sim.EpanetSimulator(self.wn)
            self.results = sim.run_sim()
        else:
            if group_id != self.dimensions:
                self.n_siesta = 0
                first_pump_in_group = self.wn.get_link(
                    self.pump_groups[group_id][0])
                if command == 0:
                    if first_pump_in_group.base_speed < self.speed_limit_high:
                        for pump_name in self.pump_groups[group_id]:
                            pump = self.wn.get_link(pump_name)
                            pump.base_speed -= self.speed_increment
                    else:
                        self.n_bump += 1
                else:
                    if first_pump_in_group.base_speed > self.speed_limit_low:
                        for pump_name in self.pump_groups[group_id]:
                            pump = self.wn.get_link(pump_name)
                            pump.base_speed -= self.speed_increment
                    else:
                        self.n_bump += 1
            else:
                self.n_siesta += 1
                if self.n_siesta == 3:
                    self.done = True
            sim = wntr.sim.EpanetSimulator(self.wn)
            self.results = sim.run_sim()
            reward = self.get_state_value()

        print('End of step')
        observation = self.get_observation()
        return observation, reward, self.done, {}

    def reset(self, training=True):
        if training:
            if self.reset_original_demands:
                self.restore_original_demands()
            else:
                self.randomize_demands()
            self.optimize_state()

        self.set_pump_speeds()

        sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = sim.run_sim()
        observation = self.get_observation()
        self.done = False
        self.steps = 0
        self.n_bump = 0
        self.n_siesta = 0
        return observation

    def optimize_state(self):
        speeds, target_val, _ = nm.minimize(
            self.reward_to_scipy, self.dimensions)
        self.optimized_speeds = speeds
        self.optimized_value = -target_val

    def fit_polynomials(self, pts_dict, degree, encapsulated=False):
        """Fitting polynomials to points stored in dict."""
        polynomials = dict()
        if encapsulated:
            for curve in pts_dict:
                polynomials[curve] = np.poly1d(np.polyfit(
                    pts_dict[curve][:, 0], pts_dict[curve][:, 1], degree))
        else:
            for curve in pts_dict:
                polynomials[curve] = np.polyfit(
                    pts_dict[curve][:, 0], pts_dict[curve][:, 1], degree)
        return polynomials

    def get_curves_dicts(self):
        head_curves = dict()
        eff_curves = dict()

        for name, curve in self.wn.curves():
            if name[0] == 'H':  # this is an H(Q) curve
                head_curves[name[1:]] = np.empty(
                    [len(curve.points), 2], dtype=np.float64)
                head_curve_points = [(np.round(point[0] * 3600, 2), point[1])
                                     for point in curve.points]

                for i, op_pnt in enumerate(head_curve_points):
                    head_curves[name[1:]][i, 0] = op_pnt[0]
                    head_curves[name[1:]][i, 1] = op_pnt[1]

            if name[0] == 'E':  # this is an E(Q) curve
                eff_curves[name[1:]] = np.empty(
                    [len(curve.points), 2], dtype=np.float64)
                efficiency_curve_points = [(np.round(point[0], 2), point[1])
                                           for point in curve.points]

                for i, op_pnt in enumerate(efficiency_curve_points):
                    eff_curves[name[1:]][i, 0] = op_pnt[0]
                    eff_curves[name[1:]][i, 1] = op_pnt[1]

        return head_curves, eff_curves

    def build_truncnorm_randomizer(self, lo, hi, mu, sigma):
        randomizer = stats.truncnorm(
            (lo-mu)/sigma, (hi-mu)/sigma, loc=mu, scale=sigma)
        return randomizer

    def set_pump_speeds(self):
        for pump_group in self.pump_groups:
            if self.reset_original_pump_speeds:
                initial_speed = 1
            else:
                initial_speed = np.random.choice(self.valid_speeds)

            for pump_name in pump_group:
                pump = self.wn.get_link(pump_name)
                pump.base_speed = initial_speed

    def set_initial_pump_speeds(self):
        for pump_group in self.pump_groups:
            initial_speed = 1
            for pump_name in pump_group:
                pump = self.wn.get_link(pump_name)
                pump.base_speed = initial_speed

    def restore_original_demands(self):
        for name, junction in self.wn.junctions():
            junction.demand_timeseries_list[0].base_value = self.demand_dict[name]

    def randomize_demands(self):
        target_sum_of_demands = self.sum_of_demands * (self.total_demand_low +
                                                       np.random.rand() * (
                                                           self.total_demand_high - self.total_demand_low))
        sum_of_random_demands = 0
        if self.seed_number:
            for name, junction in self.wn.junctions():
                junction.demand_timeseries_list[0].base_value = (self.demand_dict[name] *
                                                                 self.demand_randomizer.rvs())
                sum_of_random_demands += junction.base_demand
        else:
            for name, junction in self.wn.junctions():
                junction.demand_timeseries_list[0].base_value = (self.demand_dict[name] *
                                                                 self.demand_randomizer.rvs())
                sum_of_random_demands += junction.base_demand

        for name, junction in self.wn.junctions():
            junction.demand_timeseries_list[0].base_value *= target_sum_of_demands / \
                sum_of_random_demands

    def calculate_pump_efficiencies(self):
        for i, group in enumerate(self.pump_groups):
            pump = self.wn.get_link(group[0])
            curve_id = pump.get_pump_curve().name[1:]
            pump_head = self.results.node['head'][pump.end_node_name].iloc[0] - \
                self.results.node['head'][pump.start_node_name].iloc[0]
            eff_poli = self.nominal_E_curve_polynomials[curve_id]
            self.pump_efficiencies[i] = eff_poli(
                self.results.link['flowrate'][pump.name] * 3600 / pump.base_speed)

    def get_pump_efficiencies(self):
        pump_efficiencies = []
        for name, pump in self.wn.pumps():
            efficiencies = pump.efficiency_curve()
            pump_efficiencies.append(efficiencies)
        return pump_efficiencies

    def build_demand_dict(self):
        return {name: junc.base_demand for name, junc in self.wn.junctions()}

    def set_demands(self, series):
        for name, junction in self.wn.junctions():
            junction.demand_timeseries_list[0].base_value = series[name]

    def update_pump_speeds(self):
        for i, pump_group in enumerate(self.pump_groups):
            self.pump_speeds[i] = self.wn.get_link(pump_group[0]).base_speed
        return self.pump_speeds

    def get_pump_speeds(self):
        self.update_pump_speeds()
        return self.pump_speeds

    def get_observation(self):
        heads = (
            2 * self.results.node['head'][self.wn.junction_name_list].iloc[0].values / self.max_head) - 1
        self.update_pump_speeds()
        speeds = self.pump_speeds / self.speed_limit_high
        return np.concatenate([heads, speeds])

    def get_state_value_to_opti(self, pump_speeds):
        np.clip(a=pump_speeds,
                a_min=self.speed_limit_low,
                a_max=self.speed_limit_high,
                out=pump_speeds)

        for group_id, pump_group in enumerate(self.pump_groups):
            for pump_name in pump_group:
                pump = self.wn.get_link(pump_name)
                pump.base_speed = pump_speeds[group_id]

        # Simulate hydraulics
        sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = sim.run_sim()

        return self.get_state_value()

    def get_state_value(self):
        self.calculate_pump_efficiencies()
        pump_ok = (self.pump_efficiencies < 1).all() and (
            self.pump_efficiencies > 0).all()
        if pump_ok:
            heads = np.array([self.results.node['head'][junc].iloc[0]
                              for junc in self.wn.junction_name_list])
            invalid_heads_count = (np.count_nonzero(heads < self.head_limit_low) +
                                   np.count_nonzero(heads > self.head_limit_high))
            valid_heads_ratio = 1 - (invalid_heads_count / len(heads))

            total_demand = sum(
                [junction[1].base_demand*3600 for junction in self.wn.junctions()])

            total_tank_flow = 0

            for tank in self.wn.tanks():
                total_tank_flow += sum([abs(-self.results.link['flowrate'][link].iloc[0]
                                            * 3600) for link in self.wn.get_links_for_node(tank[0])])
            demand_to_total = total_demand / (total_demand+total_tank_flow)
            total_efficiency = np.prod(self.pump_efficiencies)

            reward = (self.rewScale[0] * total_efficiency / self.peak_total_efficiency +
                      self.rewScale[1] * valid_heads_ratio +
                      self.rewScale[2] * demand_to_total) / sum(self.rewScale)
        else:
            reward = 0
        return reward

    def reward_to_scipy(self, pump_speeds):
        """Only minimization allowed."""
        return -self.get_state_value_to_opti(pump_speeds)

    def reward_to_deap(self, pump_speeds):
        """Return should be tuple."""
        return self.get_state_value_to_opti(np.asarray(pump_speeds))

    def seed(self, seed=None):
        """Collecting seeds."""
        return [seed]
