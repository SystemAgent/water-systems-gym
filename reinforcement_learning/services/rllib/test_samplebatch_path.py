from ray.rllib.policy.sample_batch import SampleBatch
import collections
import numpy as np
import sys
from datetime import datetime
import itertools
from typing import Dict, List, Set, Union


def print_decor(func):
    '''
    Decorator for compution time of function
    '''
    def _timer(*args, **kwargs):
        start = datetime.now()
        # print('MARK')
        result = func(*args, **kwargs)
        end = datetime.now()
        # print(f"Execution time: {end - start}")
        return result
    return _timer


# SampleBatch.__init__ = print_decor(SampleBatch.__init__)
