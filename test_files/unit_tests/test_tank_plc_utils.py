import os
import yaml

import pytest

from reinforcement_learning.tank_plc_gym.utils import config_validation, find_checkpoint_path, ValidationError


def test_invalid_valve_cmd():
    """Test config validation function"""
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'test_invalid_basicTank_Rllib_DQN_valve_cmd.yaml')) as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with pytest.raises(ValidationError) as e_info:
        config_validation(hparams)

    error = str(e_info.value)
    assert error == 'Config parameter valve_cmd not in correct bounds 0:1'


def test_invalid_tank_level():
    """Test config validation function"""
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'test_invalid_basicTank_Rllib_DQN_tank_level.yaml')) as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with pytest.raises(ValidationError) as e_info:
        config_validation(hparams)

    error = str(e_info.value)
    assert error == 'Config parameter tank_level not in correct bounds 0:100'


def test_invalid_dt():
    """Test config validation function"""
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'test_invalid_basicTank_Rllib_DQN_dt.yaml')) as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with pytest.raises(ValidationError) as e_info:
        config_validation(hparams)

    error = str(e_info.value)
    assert error == 'Config parameter dt not in correct bounds 0:60'


def test_invalid_qout():
    """Test config validation function"""
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'test_invalid_basicTank_Rllib_DQN_qout.yaml')) as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with pytest.raises(ValidationError) as e_info:
        config_validation(hparams)

    error = str(e_info.value)
    assert error == 'Config parameter qout not in correct bounds 0:200'


def test_invalid_random_steps():
    """Test config validation function"""
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'test_invalid_basicTank_Rllib_DQN_random_steps.yaml')) as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with pytest.raises(ValidationError) as e_info:
        config_validation(hparams)

    error = str(e_info.value)
    assert error == 'Config parameter random_steps not in correct bounds 0:100'


def test_invalid_initial_learning_rate():
    """Test config validation function"""
    with open(os.path.join(os.path.dirname(__file__),
                           'config_test_files', 'test_invalid_basicTank_Rllib_DQN_initial_learning_rate.yaml')) as fin:
        hparams = yaml.load(fin, Loader=yaml.Loader)

    with pytest.raises(ValidationError) as e_info:
        config_validation(hparams)

    error = str(e_info.value)
    assert error == 'Config parameter initial_learning_rate not in correct bounds 0.0001:0.5'


def test_invalid_experiment():
    """Test find_checkpoint_path function"""

    with pytest.raises(ValidationError) as e_info:
        find_checkpoint_path(os.path.join(
            os.path.dirname(__file__), 'data'), 'experiment', 3)

    error = str(e_info.value)
    assert error == 'Invalid Experiment'


def test_invalid_checkpoint_number():
    """Test find_checkpoint_path function"""

    with pytest.raises(ValidationError) as e_info:
        find_checkpoint_path(os.path.join(
            os.path.dirname(__file__), 'data'), 'test_experiment', 6)

    error = str(e_info.value)
    assert error == 'Invalid checkpoint number'


def test_invalid_checkpoint():
    """Test find_checkpoint_path function"""

    with pytest.raises(ValidationError) as e_info:
        find_checkpoint_path(os.path.join(
            os.path.dirname(__file__), 'data'), 'test_experiment', 0)

    error = str(e_info.value)
    assert error == 'No checkpoint in experiment'


def test_invalid_checkpoint_file():
    """Test find_checkpoint_path function"""

    with pytest.raises(ValidationError) as e_info:
        find_checkpoint_path(os.path.join(
            os.path.dirname(__file__), 'data'), 'test_experiment', 1)

    error = str(e_info.value)
    assert error == 'Invalid Checkpoint'
