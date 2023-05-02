import argparse
import os
import glob
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
from stable_baselines.common.schedules import PiecewiseSchedule

from reinforcement_learning.pump_gym.envs.water_gym import WaterGym
from reinforcement_learning.config import PROJECT_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--params', default='anytownMaster',
                    help="Name of the YAML file.")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed for the optimization methods.")
parser.add_argument('--nproc', default=1, type=int,
                    help="Number of processes to raise.")
parser.add_argument('--tstsplit', default=20, type=int,
                    help="Ratio of scenes moved from vld to tst scene in percentage.")
args = parser.parse_args()

root_path = os.path.join(PROJECT_PATH, 'pump_gym')
experiment_path = os.path.join(root_path, 'experiments')
history_path = os.path.join(experiment_path, 'history')
parameters_path = os.path.join(
    experiment_path, 'hyperparameters', args.params+'.yaml')

with open(parameters_path, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)

scene_file_name = hparams['evaluation']['scenes'] + '.csv'
scene_path = os.path.join(
    PROJECT_PATH, 'pump_gym', 'data', scene_file_name)

history_files = [f for f in glob.glob(os.path.join(history_path, '*.h5'))]

run_id = 1
while os.path.join(history_path, args.params+str(run_id)+'_vld.h5') in history_files:
    run_id += 1
run_id = args.params+str(run_id)

validation_history_db_path = os.path.join(history_path, run_id+'_vld.h5')
best_model_path = os.path.join(experiment_path, 'models', run_id+'-best')
last_model_path = os.path.join(experiment_path, 'models', run_id+'-last')
log_path = os.path.join(experiment_path, 'tensorboard_logs')
validation_frequency = hparams['training']['total_steps'] // 25

env = WaterGym(
    wn_name=hparams['env']['water_network']+'_master',
    speed_increment=hparams['env']['speed_increment'],
    episode_len=hparams['env']['episode_length'],
    pump_groups=hparams['env']['pump_groups'],
    total_demand_low=hparams['env']['total_demand_low'],
    total_demand_high=hparams['env']['total_demand_high'],
    reset_original_demands=hparams['env']['reset_original_demands'],
    reset_original_pump_speeds=hparams['env']['reset_original_pump_speeds']
)


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=hparams['model']['layers'],
                                           dueling=True,
                                           layer_norm=False,
                                           act_fun=tf.nn.relu,
                                           feature_extraction='mlp')

def init_vldtst_history(scenes, tst=False):
    hist_header = [
        'lastReward', 'bestReward', 'worstReward',
        'nFail', 'nBump', 'nSiesta', 'nStep',
        'explorationFactor']
    for i in range(env.dimensions):
        hist_header.append('speedOfGrp'+str(i))
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
        (len(hist_index), len(hist_header)), dtype=np.float32)
    init_array.fill(np.nan)
    history = pd.DataFrame(
        init_array,
        index=hist_index,
        columns=hist_header)
    return history


def play_scenes(scenes, history, path_to_history, tst=False):
    """
    Returns:
        average_reward: float
    """
    global best_metric
    cummulated_reward = 0
    for scene_id in range(len(scenes)):
        env.set_demands(scenes.loc[scene_id])
        obs = env.reset(training=False)
        rewards = np.empty(
            shape=(env.episode_length,),
            dtype=np.float32)
        rewards.fill(np.nan)
        pump_speeds = np.empty(
            shape=(env.episode_length, env.dimensions),
            dtype=np.float32)
        while not env.done:
            act, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _ = env.step(act, training=False)
            pump_speeds[env.steps-1, :] = env.get_pump_speeds()
            rewards[env.steps-1] = reward
        cummulated_reward += reward

        if not tst:
            df_view = history.loc[step_id].loc[scene_id].copy(deep=False)
        else:
            df_view = history.loc[scene_id].copy(deep=False)
        df_view['lastReward'] = rewards[env.steps-1]
        df_view['bestReward'] = np.nanmax(rewards)
        df_view['worstReward'] = np.nanmin(rewards)
        df_view['nFail'] = np.count_nonzero(rewards == 0)
        df_view['nBump'] = env.n_bump
        df_view['nSiesta'] = env.n_siesta
        df_view['nStep'] = env.steps
        df_view['explorationFactor'] = model.exploration.value(step_id)
        for i in range(env.dimensions):
            df_view['speedOfGrp'+str(i)] = pump_speeds[env.steps-1, i]
    avg_reward = cummulated_reward / (scene_id+1)
    print('Average reward for {:} scenes: {:.3f}.'.format(
        scene_id+1, avg_reward))
    if (not tst) and (avg_reward > best_metric):
        print('Average reward improved {:.3f} --> {:.3f}.\nSaving...'
              .format(best_metric, avg_reward))
        best_metric = avg_reward
        model.save(best_model_path)
    obs = env.reset(training=True)
    history.to_hdf(path_to_history, key=run_id, mode='a')
    return avg_reward


def callback(_locals, _globals):
    global step_id, vld_history, best_metric
    step_id += 1
    if step_id % validation_frequency == 0:
        if args.tstsplit != 100:
            print('{}. step, validating.'.format(step_id))
            avg_reward = play_scenes(
                vld_scenes, vld_history, validation_history_db_path)
            if avg_reward > best_metric:
                print('Cummulated reward improved {:.3f} --> {:.3f}.\nSaving...'
                      .format(best_metric, avg_reward))
                best_metric = avg_reward
                model.save(best_model_path)
            obs = env.reset(training=True)
        vld_history.to_hdf(validation_history_db_path, key=run_id, mode='a')
    return True


step_id = 0
best_metric = 0
vldtst_scenes = pd.read_csv(scene_path, index_col=0)
if args.tstsplit:
    assert ((args.tstsplit >= 0) and (args.tstsplit <= 100))
    print('Splitting scene db to {:}% validation and {:}% test data.\n'
          .format(100-args.tstsplit, args.tstsplit))
    cut_idx = int(len(vldtst_scenes) * (100 - args.tstsplit)*0.01)
    vld_scenes = vldtst_scenes[:cut_idx].copy(deep=False)
    tst_scenes = vldtst_scenes[cut_idx:].copy(deep=False)
    tst_scenes.index = tst_scenes.index - tst_scenes.index[0]
else:
    vld_scenes = vldtst_scenes.copy(deep=False)

vld_history = init_vldtst_history(vld_scenes)
vld_history.to_hdf(validation_history_db_path, key=run_id, mode='w')

total_steps = hparams['training']['total_steps']
initial_learning_rate = hparams['training']['initial_learning_rate']
lr_schedule = PiecewiseSchedule(([
    (0, initial_learning_rate),
    (1*total_steps // 2, initial_learning_rate * .1),
    (3*total_steps // 4, initial_learning_rate * .01)
]))

model = DQN(
    policy=CustomPolicy,
    env=env,
    verbose=1,
    # learning_rate           = lr_schedule.value(step_id),
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
    n_cpu_tf_sess=args.nproc)

model.learn(
    total_timesteps=hparams['training']['total_steps'],
    log_interval=hparams['training']['total_steps'] // 50,
    callback=callback,
    tb_log_name=args.params)
model.save(last_model_path)

if args.tstsplit:
    print('End of training, testing.\n')
    tst_history = init_vldtst_history(tst_scenes, tst=True)
    test_history_path = os.path.join(history_path, run_id+'_tst.h5')
    tst_history.to_hdf(test_history_path, key=run_id, mode='w')
    play_scenes(tst_scenes, tst_history, test_history_path, tst=True)
