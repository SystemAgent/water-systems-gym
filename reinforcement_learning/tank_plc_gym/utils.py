import os
from functools import reduce
from operator import getitem

import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo.appo as appo
from ray.tune.callback import Callback
import mlflow

from reinforcement_learning.services.mlflow.utils import MlflowClient
from reinforcement_learning.config import MLFLOW_USER, MLFLOW_PASSWORD, MLFLOW_TRACKING_URI
from reinforcement_learning.services.rllib import DQNTrainer
from reinforcement_learning.services.rllib import SampleBatch


matplotlib.use('Agg')
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USER
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD

agents_config = {'ppo_disc': {'config': ppo.DEFAULT_CONFIG.copy(),
                              'agent': ppo.PPOTrainer},
                 'ppo_cont': {'config': ppo.DEFAULT_CONFIG.copy(),
                              'agent': ppo.PPOTrainer},
                 'ddpg': {'config': ddpg.DEFAULT_CONFIG.copy(),
                          'agent': ddpg.DDPGTrainer},
                 'appo': {'config': appo.DEFAULT_CONFIG.copy(),
                          'agent': appo.APPOTrainer},
                 'dqn': {'config': dqn.DEFAULT_CONFIG.copy(),
                         'agent': DQNTrainer},
                 }


class CheckpointCallback(Callback):
    def __init__(self, data_path, env, pattern_scenes, select_env, hparams):
        """[summary]

        :param data_path: [description]
        :type data_path: [type]
        :param env: [description]
        :type env: [type]
        :param pattern_scenes: [description]
        :type pattern_scenes: [type]
        :param select_env: [description]
        :type select_env: [type]
        :param hparams: [description]
        :type hparams: [type]
        """
        self.data_path = data_path
        self.env = env
        self.pattern_scenes = pattern_scenes
        self.validation_number = 0
        self.hparams = hparams
        self.select_env = select_env
        super().__init__()

    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result")

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        """[summary]

        :param iteration: [description]
        :type iteration: [type]
        :param trials: [description]
        :type trials: [type]
        :param trial: [description]
        :type trial: [type]
        :param checkpoint: [description]
        :type checkpoint: [type]
        """
        print(f"CHECHPOINTING")
        validation_path = os.path.join(
            self.data_path, 'checkpoint_' + 'rllib' + str(self.validation_number))
        if not os.path.exists(validation_path):
            os.mkdir(validation_path)

        agent, _, _ = create_agent(select_env=self.select_env, hparams=self.hparams,
                                   agent_type=self.hparams['model']['type'], workers=1)
        agent.restore(checkpoint.value)

        scene_evaluation(pattern_scenes=self.pattern_scenes, env=self.env,
                         agent=agent, validation_path=validation_path)

        self.validation_number += 1


class ValidationError(Exception):
    pass


def find_checkpoint_path(data_path, experiment_name, checkpoint_number):
    """A function to find the path for saving checkpoint data

    :param data_path: [description]
    :type data_path: [type]
    :param experiment_name: [description]
    :type experiment_name: [type]
    :param checkpoint_number: [description]
    :type checkpoint_number: [type]
    :raises ValidationError: [description]
    :raises ValidationError: [description]
    :raises ValidationError: [description]
    :raises ValidationError: [description]
    :return: [description]
    :rtype: [type]
    """
    experiment_path = os.path.join(data_path, experiment_name)
    if not os.path.exists(experiment_path):
        raise ValidationError('Invalid Experiment')

    checkpoint_folder = None
    for file in os.listdir(experiment_path):
        if os.path.isdir(os.path.join(experiment_path, file)) and (str(checkpoint_number) in file):
            checkpoint_folder = os.path.join(experiment_path, file)

    if not checkpoint_folder:
        raise ValidationError('Invalid checkpoint number')

    checkpoint_path = None
    checkpoint_number = None
    for file in os.listdir(checkpoint_folder):
        if os.path.isdir(os.path.join(checkpoint_folder, file)) and ('checkpoint' in file):
            checkpoint_path = os.path.join(checkpoint_folder, file)
            checkpoint_number = int(file.split('_')[1])

    if not checkpoint_path:
        raise ValidationError('No checkpoint in experiment')

    checkpoint_path = os.path.join(
        checkpoint_path, 'checkpoint-' + str(checkpoint_number))

    if not os.path.exists(checkpoint_path):
        raise ValidationError('Invalid Checkpoint')

    return checkpoint_path


def create_agent(select_env, hparams, agent_type=None, **kwargs):
    """ A function to initiate the agent

    :param select_env: [A gym.env to be used with the agent]
    :type select_env: [type]
    :param hparams: [description]
    :type hparams: [type]
    :param agent_type: [Types of RL algorithms implemented e.g 'ppo_disc', 'dqn', 'ddpg', 'ppo_cont', 'appo']
    :type agent_type: [type], optional
    :return: [description]
    :rtype: [type]
    """
    if agent_type not in ['ppo_disc', 'dqn', 'ddpg', 'ppo_cont', 'appo']:
        return 'Agent type unknown.'

    config = agents_config[agent_type]['config']
    config['log_level'] = 'WARN'

    # set number of workers
    if 'num_workers' in hparams['training'].keys():
        config['num_workers'] = hparams['training']['num_workers']
    if 'workers' in kwargs.keys():
        config['num_workers'] = kwargs['workers']

    # set learning rate
    if 'initial_learning_rate' in hparams['training'].keys():
        config['lr'] = hparams['training']['initial_learning_rate']

    # set batch_size
    if 'batch_size' in hparams['training'].keys():
        config['train_batch_size'] = hparams['training']['batch_size']

    # set number of timesteps for every iteration
    if 'timesteps_per_iteration' in hparams['training'].keys():
        config['timesteps_per_iteration'] = hparams['training']['timesteps_per_iteration']

    # set learning starts(steps until weight update)
    if 'learning_starts' in hparams['training'].keys():
        config['learning_starts'] = hparams['training']['learning_starts']

    if 'vf_clip_param' in hparams['training'].keys():
        config['vf_clip_param'] = hparams['training']['vf_clip_param']

    # set NN architecture
    if 'layers' in hparams['model'].keys():
        config['model']['fcnet_hiddens'] = hparams['model']['layers']

    agent = agents_config[agent_type]['agent'](config, env=select_env)
    return agent, agents_config[agent_type]['agent'], config


def subplot(df, axs, ax_x, ax_y, column_name, xlabel, ylabel, title):
    """ A function to add a subplot to the main training vizualization

    :param df: [description]
    :type df: [type]
    :param axs: [description]
    :type axs: [type]
    :param ax_x: [description]
    :type ax_x: [type]
    :param ax_y: [description]
    :type ax_y: [type]
    :param column_name: [description]
    :type column_name: [type]
    :param xlabel: [description]
    :type xlabel: [type]
    :param ylabel: [description]
    :type ylabel: [type]
    :param title: [description]
    :type title: [type]
    """
    df[column_name].plot(ax=axs[ax_x, ax_y])
    axs[ax_x, ax_y].set(xlabel=xlabel, ylabel=ylabel)
    axs[ax_x, ax_y].set_title(title)


def main_visualization(result_dataframes, validation_path, lib, split_visualization=False):
    """ A function to plot the datta available from the Agent training

    :param result_dataframes: [The data frame containing the result data from training]
    :type result_dataframes: [pdndas.DataFrame]
    :param validation_path: [description]
    :type validation_path: [type]
    :param lib: [description]
    :type lib: [type]
    :param split_visualization: [description], defaults to False
    :type split_visualization: bool, optional
    """
    rewards = []
    scenes = []
    lib = ''
    for index, df in enumerate(result_dataframes):
        fig, axs = plt.subplots(
            2, 3, figsize=(21, 12))
        subplot(df, axs, 0, 0, 'rewards', 'Time', 'Reward', 'Reward')
        subplot(df, axs, 0, 1, 'level', 'Time', 'Tank Level', 'Tank Level')
        subplot(df, axs, 0, 2, 'qout', 'Time', 'Demand', 'Demand')
        subplot(df, axs, 1, 0, 'flow_rate', 'Time', 'Flowrate', 'Flowrate')
        subplot(df, axs, 1, 1, 'command', 'Time', 'Commands', 'Command')
        np.abs(df['level'] - 50).plot(ax=axs[1, 2])
        axs[1, 2].set(xlabel='Time', ylabel='Error')
        axs[1, 2].set_title('Error of tank level')
        plt.savefig(validation_path + '/control_scene_' +
                    lib + str(index) + '.png')
        # plt.show()
        plt.close('all')


def metrics_validations(series, min=None, max=None):
    """ Compare metrics with expected distribution values

    :return: [description]
    :rtype: [type]
    """
    x = series.mean()
    if x < min or x > max:
        return False
    return True


def agent_performance_validation(df, result, scene_id):
    """ A function for agent performance testing and validation.

    :param df: [description]
    :type df: [pandas.DataFrame]
    :param result: [description]
    :type result: [pandas.DataFrame]
    :param scene_id: [description]
    :type scene_id: [type]
    :return: [description]
    :rtype: [pandas.DataFrame]
    """
    # first 100 rewards the agent is still reaching the set point
    result['rewards_scene_minimal'][f'scene_{scene_id}'] = metrics_validations(
        df['rewards'][100:], 0, np.inf)
    result[f'level_scene_minimal'][f'scene_{scene_id}'] = metrics_validations(
        np.abs(df['level'] - 50)[100:], 0, 10)

    result[f'rewards_scene_strict'][f'scene_{scene_id}'] = metrics_validations(
        df['rewards'][100:], 9, np.inf)
    result[f'level_scene_strict'][f'scene_{scene_id}'] = metrics_validations(
        np.abs(df['level'] - 50)[100:], 0, 5)

    return result


def scene_evaluation(pattern_scenes, env, agent, validation_path):
    """ A function to evaluate scenes, part of the training process

    :param pattern_scenes: [description]
    :type pattern_scenes: [type]
    :param env: [description]
    :type env: [type]
    :param agent: [description]
    :type agent: [type]
    :param validation_path: [description]
    :type validation_path: [type]
    """
    result_dataframes = []
    performance_result = {
        'rewards_scene_minimal': {},
        'level_scene_minimal': {},
        'rewards_scene_strict': {},
        'level_scene_strict': {}
    }
    for scene_id in range(len(pattern_scenes)):
        obs = env.reset()
        env.set_pattern(
            list(pattern_scenes.loc[scene_id+1].to_numpy()))
        sum_reward = 0

        rewards = np.empty(
            shape=env.episode_length,
            dtype=np.float64)
        rewards.fill(np.nan)
        valve_commands = np.empty(
            shape=env.episode_length,
            dtype=np.float64)
        tank_levels = np.empty(
            shape=env.episode_length,
            dtype=np.float64)
        qout = np.empty(
            shape=env.episode_length,
            dtype=np.float64)
        flow_rate = np.empty(
            shape=env.episode_length,
            dtype=np.float64)
        actions = np.empty(
            shape=env.episode_length,
            dtype=np.float64)

        # evaluate agent
        while not env.done:
            act = agent.compute_action(obs, explore=False)
            obs, reward, _, _ = env.step(act, debug=False)
            valve_commands[env.steps-1] = env.tank.valve_cmd
            tank_levels[env.steps-1] = env.tank.tank_level
            rewards[env.steps-1] = reward
            qout[env.steps-1] = env.tank.qout_next
            flow_rate[env.steps-1] = env.tank.input_flow_rate
            actions[env.steps-1] = act
        sum_reward += reward

        # save episodic data
        df = pd.DataFrame(
            {'level': tank_levels, 'command': valve_commands, 'rewards': rewards, 'qout': qout, 'actions': actions, 'flow_rate': flow_rate})

        # first 100 rewards the agent is still reaching the set point
        agent_performance_validation(
            df=df, result=performance_result, scene_id=scene_id)

        df.to_csv(validation_path + '/RLscenes' +
                  '_' + str(scene_id) + '.csv')
        result_dataframes.append(df)

    pd.DataFrame(performance_result).to_csv(
        validation_path + '/result_summary' + '.csv')

    main_visualization(result_dataframes, validation_path, lib='rllib')


def get_nested_item(data, keys):
    """
    :param data: [description]
    :type data: [type]
    :param keys: [description]
    :type keys: [type]
    :return: [description]
    :rtype: [type]
    """
    return reduce(getitem, keys, data)


def log_rllib_metrics_mlflow(result, artifact_paths, model_type, experiment_name='RL_models', run_name='rllib'):
    """ A function for logging the experiments and acconmpanying data to MLflow

    :param result: [description]
    :type result: [pandas.DataFrame]
    :param artifact_paths: [description]
    :type artifact_paths: [type]
    :param model_type: type of agent RL model to be used
    :type model_type: str, optional
    :param experiment_name: name of the current experiment defaults to 'RL_models'
    :type experiment_name: str, optional
    :param run_name: name of the current run, defaults to 'rllib'
    :type run_name: str, optional
    """
    client = MlflowClient()

    # get experiment and check if it exists
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment = client.create_experiment(experiment_name)

    metrics_dqn = [['episode_reward_max'], ['episode_reward_mean'],
                   ['episode_reward_min'], ['info', 'learner',
                                            'default_policy', 'learner_stats', 'cur_lr'],
                   ['info', 'learner', 'default_policy', 'learner_stats', 'mean_q'],
                   ['info', 'learner', 'default_policy', 'learner_stats', 'min_q'],
                   ['info', 'learner', 'default_policy', 'learner_stats', 'max_q'],
                   ['info', 'learner', 'default_policy', 'learner_stats', 'mean_td_error']]

    metrics_ppo = [['episode_reward_max'], ['episode_reward_mean'],
                   ['episode_reward_min'], ['info', 'learner',
                                            'default_policy', 'learner_stats', 'kl'],
                   ['info', 'learner', 'default_policy',
                       'learner_stats', 'cur_kl_coeff'],
                   ['info', 'learner', 'default_policy',
                       'learner_stats', 'entropy'],
                   ['info', 'learner', 'default_policy', 'learner_stats', 'entropy_coeff']]
    params = [['episode_len_mean'], ['time_total_s'],
              ['config', 'num_workers'], ['config', 'train_batch_size'],
              ['config', 'model', 'fcnet_hiddens'], ['training_iteration']]

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        if model_type == 'dqn':
            for metric in metrics_dqn:
                for index, item in enumerate(result):
                    client.log_metric(run.info.run_id, metric[-1],
                                      get_nested_item(item, metric), step=index)
        elif model_type == 'ppo_disc' or 'ppo_cont':
            for metric in metrics_ppo:
                for index, item in enumerate(result):
                    client.log_metric(run.info.run_id, metric[-1],
                                      get_nested_item(item, metric), step=index)

        for param in params:
            client.log_param(run.info.run_id, param[-1],
                             get_nested_item(result[-1], param))

        for artifact_path in artifact_paths:
            client.log_artifact(run.info.run_id, artifact_path)


def check_param_interval(param, param_name, min_value, max_value):
    """ A function to validate on parameters intervals and deviations from those.

    :param param: [description]
    :type param: [type]
    :param param_name: [description]
    :type param_name: [type]
    :param min_value: [description]
    :type min_value: [type]
    :param max_value: [description]
    :type max_value: [type]
    :raises ValidationError: [description]
    """
    if param < min_value or param > max_value:
        raise ValidationError(
            f'Config parameter {param_name} not in correct bounds {min_value}:{max_value}')


def check_config_parameter(hparams, type_param, param, min_value, max_value):
    """A function to validate on config parameters.

    :param hparams: [description]
    :type hparams: [type]
    :param type_param: [description]
    :type type_param: [type]
    :param param: [description]
    :type param: [type]
    :param min_value: [description]
    :type min_value: [type]
    :param max_value: [description]
    :type max_value: [type]
    """
    if param in hparams[type_param].keys():
        check_param_interval(
            param=hparams[type_param][param], param_name=param, min_value=min_value, max_value=max_value)


def config_validation(hparams):
    """A function to validate on config parameters

    :param hparams: [description]
    :type hparams: [type]
    """
    check_config_parameter(hparams, 'env', 'dt', 0, 60)
    check_config_parameter(hparams, 'env', 'valve_cmd', 0, 1)
    check_config_parameter(hparams, 'env', 'tank_level', 0, 100)
    check_config_parameter(hparams, 'env', 'qout', 0, 200)
    check_config_parameter(hparams, 'env', 'random_steps', 0, 100)
    check_config_parameter(
        hparams, 'training', 'initial_learning_rate', 0.0001, 0.5)
    # for potential production models at least 50000 training steps are necessary
    # check_config_parameter(
    #     hparams, 'model', 'number_of_iterations', 50000, 10000000)


def observation_validation(obs):
    """A function to validate on observation part of the training process

    :param obs: [description]
    :type obs: [type]
    """
    check_param_interval(
        param=obs[0], param_name='tank_level', min_value=0, max_value=100)
    check_param_interval(
        param=obs[1], param_name='valve_cmd', min_value=0, max_value=1)
    check_param_interval(
        param=obs[2], param_name='qout', min_value=0, max_value=200)


def save_weights(agent, path):
    """ A function to save the weights of the trained agent

    :param agent: [description]
    :type agent: [type]
    :param path: [description]
    :type path: [type]
    """
    new_weghts = {name: value.tolist() for name, value in agent.get_weights()[
        'default_policy'].items()}
    with open(path, "w") as f:
        json.dump(new_weghts, f, indent=4)


def function_timer(func):
    """ Decorator for compution time of function
    :return: [description]
    :rtype: [type]
    """
    def _timer(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f"Execution time: {end - start}")
        return result
    return _timer
