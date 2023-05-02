import os
import yaml

import pytest
import pandas as pd

from reinforcement_learning.tank_plc_gym.utils import agent_performance_validation, find_checkpoint_path, ValidationError


def test_valid_agent_performance_minimal():
    """Test agent performance function"""
    scene_id = 3
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  'data', 'test_checkpoint_scenes', f'RLscenes_{scene_id}.csv'), index_col=0)

    performance_result = {
        'rewards_scene_minimal': {},
        'level_scene_minimal': {},
        'rewards_scene_strict': {},
        'level_scene_strict': {}
    }
    agent_performance_validation(
        df=df, result=performance_result, scene_id=scene_id)

    assert performance_result['rewards_scene_minimal'][f'scene_{scene_id}'] == True
    assert performance_result['level_scene_minimal'][f'scene_{scene_id}'] == True


def test_valid_agent_performance_strict():
    """Test config validation function"""
    scene_id = 1
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  'data', 'test_checkpoint_scenes', f'RLscenes_{scene_id}.csv'), index_col=0)

    performance_result = {
        'rewards_scene_minimal': {},
        'level_scene_minimal': {},
        'rewards_scene_strict': {},
        'level_scene_strict': {}
    }
    agent_performance_validation(
        df=df, result=performance_result, scene_id=scene_id)

    assert performance_result['rewards_scene_strict'][f'scene_{scene_id}'] == True
    assert performance_result['level_scene_strict'][f'scene_{scene_id}'] == True


def test_invalid_agent_performance_strict():
    """Test config validation function"""
    scene_id = 0
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  'data', 'test_checkpoint_scenes', f'RLscenes_{scene_id}.csv'), index_col=0)

    performance_result = {
        'rewards_scene_minimal': {},
        'level_scene_minimal': {},
        'rewards_scene_strict': {},
        'level_scene_strict': {}
    }
    agent_performance_validation(
        df=df, result=performance_result, scene_id=scene_id)

    assert performance_result['rewards_scene_strict'][f'scene_{scene_id}'] == False
    assert performance_result['level_scene_strict'][f'scene_{scene_id}'] == False
    assert performance_result['rewards_scene_minimal'][f'scene_{scene_id}'] == True
    assert performance_result['level_scene_minimal'][f'scene_{scene_id}'] == True


def test_invalid_agent_performance_minimal():
    """Test config validation function"""
    scene_id = 4
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                  'data', 'test_checkpoint_scenes', f'RLscenes_{scene_id}.csv'), index_col=0)

    performance_result = {
        'rewards_scene_minimal': {},
        'level_scene_minimal': {},
        'rewards_scene_strict': {},
        'level_scene_strict': {}
    }
    agent_performance_validation(
        df=df, result=performance_result, scene_id=scene_id)

    assert performance_result['rewards_scene_minimal'][f'scene_{scene_id}'] == False
    assert performance_result['level_scene_minimal'][f'scene_{scene_id}'] == False
