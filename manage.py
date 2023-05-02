#!/usr/bin/env python
import sys
import os
from subprocess import call
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import click

from reinforcement_learning import config


@click.command(context_settings={'ignore_unknown_options': True})
@click.option('--env', default='testing',
              help='which testing env to use(testing or gitlab)')
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def test(env, args):
    """Run tests via pytest.

    This command catches all arguments and options and pass them to pytest
    running from the root path with ENV set to testing and applied test shims.

    Run 'pytest --help' for how to use pytest.
    """
    os.environ['ENV'] = env
    os.environ['PYTHONPATH'] = os.path.join(
        config.ROOT_PATH, 'test_files', 'shims')
    sys.exit(call(['pytest'] + list(args), cwd=os.path.join(config.ROOT_PATH)))


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    pass


if __name__ == '__main__':
    cli.add_command(test)
    cli()
