import gym
import numpy as np
import pytest
from gym.error import Error


def test_initialization_types_disc(tank_gym_env_disc):
    """Test env with discrete Action space and Box Observation space."""
    env_disc = tank_gym_env_disc[0]
    if not isinstance(env_disc.observation_space, gym.spaces.Box):
        raise Error(
            'TankGym environment with PPO_Discrete or DQN model type'
            ' requires an observation space of type gym.spaces.Box')
    assert type(env_disc.observation_space) == gym.spaces.Box

    if not isinstance(env_disc.action_space, gym.spaces.Discrete):
        raise Error(
            'TankGym environment with PPO_Discrete or DQN model type'
            ' requires an action space of type gym.spaces.Discrete')
    assert type(env_disc.action_space) == gym.spaces.Discrete


def test_initialization_types_cont(tank_gym_env_cont):
    """Test env with Box Action space and Box Observation space."""
    env_cont = tank_gym_env_cont[0]
    if not isinstance(env_cont.observation_space, gym.spaces.Box):
        raise Error(
            'TankGym environment with PPO_Continuous or APPO model type'
            ' requires an observation space of type gym.spaces.Box')
    assert type(env_cont.observation_space) == gym.spaces.Box

    if not isinstance(env_cont.action_space, gym.spaces.Box):
        raise Error(
            'TankGym environment with PPO_Continuous or APPO model type'
            ' requires an action space of type gym.spaces.Discrete')
    assert type(env_cont.action_space) == gym.spaces.Box


def test_step_no_action_cont(tank_gym_env_cont):
    action = []
    with pytest.raises(Error) as e_info:
        tank_gym_env_cont[0].step(action)

    error = str(e_info.value)
    assert error == 'Action out of the environment action space provided.' \
                    'Step function requires an action from the env.action_space.'


def test_step_wrong_action_cont(tank_gym_env_cont):
    action = [1.01]
    with pytest.raises(Error) as e_info:
        tank_gym_env_cont[0].step(action)

    error = str(e_info.value)
    assert error == 'Action out of the environment action space provided.' \
                    'Step function requires an action from the env.action_space.'


def test_step_wrong_negative_action_cont(tank_gym_env_cont):
    action = [-1.01]
    with pytest.raises(Error) as e_info:
        tank_gym_env_cont[0].step(action)

    error = str(e_info.value)
    assert error == 'Action out of the environment action space provided.' \
                    'Step function requires an action from the env.action_space.'


def test_step_no_action_disc(tank_gym_env_disc):
    action = []
    with pytest.raises(Error) as e_info:
        tank_gym_env_disc[0].step(action)

    error = str(e_info.value)
    assert error == 'Action out of the environment action space provided.' \
                    'Step function requires an action from the env.action_space.'


def test_step_wrong_action_disc(tank_gym_env_disc):
    action = 1000
    with pytest.raises(Error) as e_info:
        tank_gym_env_disc[0].step(action)

    error = str(e_info.value)
    assert error == 'Action out of the environment action space provided.' \
                    'Step function requires an action from the env.action_space.'


def test_step_wrong_negative_action_disc(tank_gym_env_disc):
    action = -1
    with pytest.raises(Error) as e_info:
        tank_gym_env_disc[0].step(action)

    error = str(e_info.value)
    assert error == 'Action out of the environment action space provided.' \
                    'Step function requires an action from the env.action_space.'


def test_disc_lazy_increment(tank_gym_env_cont):
    action = 'action'
    with pytest.raises(Error) as e_info:
        tank_gym_env_cont[0].disc_lazy_increment(action)

    error = str(e_info.value)
    assert error == 'Action space must be Discrete for usage of this function.'


def test_cont_lazy_increment(tank_gym_env_disc):
    action = 'action'
    with pytest.raises(Error) as e_info:
        tank_gym_env_disc[0].continuous_lazy_increment(action)

    error = str(e_info.value)
    assert error == 'Action space must be Box for usage of this function.'


def test_continuous_differential_increment(tank_gym_env_cont):
    action = [110]
    with pytest.raises(IndexError) as e_info:
        tank_gym_env_cont[0].continuous_differntial_increment(
            action, max_increment=2)

    error = str(e_info.value)
    assert error == 'action out of range'


def test_get_observation(tank_gym_env_disc):
    assert type(tank_gym_env_disc[0].get_observation()) == np.ndarray
    assert tank_gym_env_disc[0].get_observation().shape == (3,)


def test_reset(tank_gym_env_disc, tank_gym_env_cont):
    tank_gym_env_disc[0].reset(testing=True)
    obs = tank_gym_env_disc[0].get_observation()
    assert (obs.round(3) == np.array([0, 0.5, 0])).all()

    tank_gym_env_cont[0].reset(testing=True)
    obs = tank_gym_env_cont[0].get_observation()
    assert (obs.round(3) == np.array([50, 0.5, 0])).all()


def test_valid_disc_step(tank_gym_env_disc):
    tank_gym_env_disc[0].reset(testing=True)
    action = 5
    obs = tank_gym_env_disc[0].step(action)
    assert (obs[0].round(3) == np.array([0.476, 0.8, 22])).all()


def test_valid_cont_step(tank_gym_env_cont):
    tank_gym_env_cont[0].reset(testing=True)
    action = [0.9]
    obs = tank_gym_env_cont[0].step(action)
    assert (obs[0].round(3) == np.array([50.556, 0.9, 22])).all()


# TODO test_get_state_value
