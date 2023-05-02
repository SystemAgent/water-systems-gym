import argparse
import os
import glob

import tensorflow as tf
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from stable_baselines.common.schedules import PiecewiseSchedule
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.deepq import DQN

from stable_baselines.deepq.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy as FFPolicy

from reinforcement_learning.tank_plc_gym.utils import main_visualization
from reinforcement_learning.config import PROJECT_PATH
from reinforcement_learning.tank_plc_gym.envs.tank_gym import TankGym


tf.get_logger().setLevel('INFO')

parser = argparse.ArgumentParser()
parser.add_argument('--params', default='basicTank_stb_DQN',
                    help="Name of the YAML file.")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed for the optimization methods.")
parser.add_argument('--nproc', default=1, type=int,
                    help="Number of processes to raise.")
parser.add_argument('--tstsplit', default=20, type=int,
                    help="Ratio of scenes moved from vld to tst scene in percentage.")
args = parser.parse_args()

root_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym')
experiment_path = os.path.join(root_path, 'experiments')
history_path = os.path.join(experiment_path, 'history')
parameters_path = os.path.join(
    experiment_path, 'hyperparameters', args.params+'.yaml')

with open(parameters_path, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)

scene_file_name = hparams['evaluation']['scenes'] + '.csv'
scene_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym', 'data', scene_file_name)
data_path = os.path.join(
    PROJECT_PATH, 'tank_plc_gym', 'data')

history_files = [f for f in glob.glob(os.path.join(history_path, '*.h5'))]

run_id = 1
while os.path.join(history_path, args.params+str(run_id)+'_vld.h5') in history_files:
    run_id += 1
run_id = args.params+str(run_id)

validation_history_db_path = os.path.join(history_path, run_id+'_vld.h5')
best_model_path = os.path.join(experiment_path, 'models', run_id+'-best')
last_model_path = os.path.join(experiment_path, 'models', run_id+'-last')
log_path = os.path.join(experiment_path, 'tensorboard_logs')
validation_frequency = hparams['training']['total_steps'] // 5


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


class CustomPolicyDQN(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyDQN, self).__init__(*args, **kwargs,
                                              layers=hparams['model']['layers'],
                                              dueling=True,
                                              layer_norm=False,
                                              act_fun=tf.nn.relu,
                                              feature_extraction='mlp')


class CustomPolicyPPO(FFPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyPPO, self).__init__(*args, **kwargs,
                                              net_arch=[dict(pi=hparams['model']['pi_layer'],
                                                             vf=hparams['model']['vi_layer'])],
                                              act_fun=tf.nn.relu,
                                              feature_extraction='mlp')


def init_vldtst_history(scenes, tst=False):
    """ Initialise validation history .

    Args:
        scenes

    Returns:
        history: pd.DataFrame

    Raises:

    """
    # print('INIT')
    hist_header = [
        'lastReward', 'bestReward', 'worstReward',
        'nFail', 'nStep', 'explorationFactor',
        'tank_levels', 'valve_commands']
    scene_ids = np.arange(len(scenes))
    step_ids = np.arange(
        validation_frequency,
        hparams['training']['total_steps']+1,
        validation_frequency)
    if not tst:
        hist_index = pd.MultiIndex.from_product(
            [step_ids, scene_ids],
            names=['step_id', 'scene_id'])
    else:
        hist_index = pd.Index(scene_ids, name='scene_id')
    init_array = np.empty(
        (len(hist_index), len(hist_header)), dtype=np.float64)
    init_array.fill(np.nan)
    history = pd.DataFrame(
        init_array,
        index=hist_index,
        columns=hist_header)
    print('End init')
    return history


def play_scenes(scenes, history, path_to_history, validation_number, tst=False):
    """ Function for playing passed scenes.

    :param scenes: [description]
    :type scenes: [type]
    :param history: [description]
    :type history: [type]
    :param path_to_history: [description]
    :type path_to_history: [type]
    :param validation_number: [description]
    :type validation_number: [type]
    :param tst: [description], defaults to False
    :type tst: bool, optional
    :return: [description]
    :rtype: [type]
    """
    print('Playing scenes')
    global best_metric
    cummulated_reward = 0
    result_dataframes = []
    for scene_id in range(len(scenes)):
        env.set_pattern(scenes.loc[scene_id+1])
        obs = env.reset()
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

        while not env.done:
            act, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _ = env.step(act)
            valve_commands[env.steps-1] = env.tank.valve_cmd
            tank_levels[env.steps-1] = env.tank.tank_level
            rewards[env.steps-1] = reward
            qout[env.steps-1] = env.tank.qout_next
            flow_rate[env.steps-1] = env.tank.input_flow_rate
            actions[env.steps-1] = act

        validation_path = os.path.join(
            data_path, 'validation_' + 'stb' + str(validation_number))
        if not os.path.exists(validation_path):
            os.mkdir(validation_path)

        df = pd.DataFrame(
            {'level': tank_levels, 'flow_rate': flow_rate, 'command': valve_commands, 'rewards': rewards, 'qout': qout, 'actions': actions})
        df.to_csv(validation_path + '/scene_' + 'stb' + str(scene_id) + '.csv')
        cummulated_reward += df['rewards'].mean()
        result_dataframes.append(df)

        if not tst:
            df_view = history.loc[step_id].loc[scene_id].copy(deep=False)
        else:
            df_view = history.loc[scene_id].copy(deep=False)
        df_view['lastReward'] = rewards[env.steps-1]
        df_view['bestReward'] = np.nanmax(rewards)
        df_view['worstReward'] = np.nanmin(rewards)
        df_view['nFail'] = np.count_nonzero(rewards == 0)
        df_view['nStep'] = env.steps
        # df_view['explorationFactor'] = model.exploration.value(step_id)
        df_view['tank_levels'] = tank_levels[env.steps-1]
        df_view['valve_commands'] = valve_commands[env.steps-1]

    main_visualization(result_dataframes, validation_path, lib='stb')

    avg_reward = cummulated_reward / (scene_id+1)
    print('Average reward for {:} scenes: {:.3f}.'.format(
        scene_id+1, avg_reward))
    if (not tst) and (avg_reward > best_metric):
        print('Average reward improved {:.3f} --> {:.3f}.\nSaving...'
              .format(best_metric, avg_reward))
        best_metric = avg_reward
        model.save(best_model_path)
    obs = env.reset()
    history.to_hdf(path_to_history, key=run_id, mode='a')
    return avg_reward


def callback(_locals, _globals):
    """[summary]

    :param _locals: [description]
    :type _locals: [type]
    :param _globals: [description]
    :type _globals: [type]
    :return: [description]
    :rtype: [type]
    """
    global step_id, vld_history, best_metric, validation_number
    step_id += 1
    if step_id % 1000 == 0:
        print(f'Done: {step_id} steps')
    if step_id % validation_frequency == 0:
        if args.tstsplit != 100:
            print('{}. step, validating.'.format(step_id))
            avg_reward = play_scenes(
                vld_scenes, vld_history, validation_history_db_path, validation_number)
            print(avg_reward)
            if avg_reward > best_metric:
                print('Cummulated reward improved {:.3f} --> {:.3f}.\nSaving...'
                      .format(best_metric, avg_reward))
                best_metric = avg_reward
                model.save(best_model_path)
            obs = env.reset()
            validation_number += 1
        # vld_history.to_hdf(validation_history_db_path, key=run_id, mode='a')
    return True


step_id = 0
best_metric = 0
vld_scenes = pd.read_csv(scene_path, index_col=0)
validation_number = 0

vld_history = init_vldtst_history(vld_scenes)
vld_history.to_hdf(validation_history_db_path, key=run_id, mode='w')

total_steps = hparams['training']['total_steps']
initial_learning_rate = hparams['training']['initial_learning_rate']
lr_schedule = PiecewiseSchedule(([
    (0, initial_learning_rate),
    (1*total_steps // 2, initial_learning_rate * .1),
    (3*total_steps // 4, initial_learning_rate * .01)
]))

if hparams['model']['type'] in ['ppo', 'ppo_cont']:
    model = PPO1(
        policy=CustomPolicyPPO,
        env=env,
        verbose=0,
        adam_epsilon=0.0001,
        gamma=hparams['training']['gamma'],
        tensorboard_log=log_path,
        full_tensorboard_log=True,
        seed=args.seed,
        n_cpu_tf_sess=8)

if hparams['model']['type'] == 'dqn':
    model = DQN(
        policy=CustomPolicyDQN,
        env=env,
        verbose=1,
        learning_rate=initial_learning_rate,
        buffer_size=hparams['training']['buffer_size'],
        gamma=hparams['training']['gamma'],
        batch_size=hparams['training']['batch_size'],
        learning_starts=hparams['training']['learning_starts'],
        exploration_fraction=.95,
        exploration_final_eps=.0,
        param_noise=False,
        prioritized_replay=False,
        tensorboard_log=log_path,
        full_tensorboard_log=True,
        seed=args.seed,
        n_cpu_tf_sess=8)

model.learn(
    total_timesteps=hparams['training']['total_steps'],
    log_interval=hparams['training']['total_steps'] // 50,
    callback=callback,
    tb_log_name=args.params)
model.save(last_model_path)

validation_number = 'final'
if args.tstsplit:
    print('End of training, testing.\n')
    tst_history = init_vldtst_history(vld_scenes, tst=True)
    test_history_path = os.path.join(history_path, run_id+'_tst.h5')
    tst_history.to_hdf(test_history_path, key=run_id, mode='w')
    play_scenes(vld_scenes, tst_history, test_history_path,
                validation_number, tst=True)
