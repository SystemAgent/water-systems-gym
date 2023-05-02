"""Define important project constants."""
from base64 import b64decode
import os
from uuid import uuid4
import tempfile

from dotenv import load_dotenv


# Notable paths needed inside the project.
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.join(PROJECT_PATH, '..')

env_file = '.prod.env'
DEBUG = False
TESTING = False
SENTRY_DSN = None

if os.environ.get('ENV') == 'dev':
    DEBUG = True
elif os.environ.get('ENV') == 'testing':
    env_file = '.test.env'
    TESTING = True
    DEBUG = True
elif os.environ.get('ENV') == 'gitlab':
    env_file = '.gitlab.env'
    TESTING = True
    DEBUG = True

load_dotenv(os.path.join(ROOT_PATH, env_file), override=True)

# mlflow
MLFLOW_USER = os.environ.get('MLFLOW_USER', '')
MLFLOW_PASSWORD = os.environ.get('MLFLOW_PASSWORD', '')
MLFLOW_ARTIFACT_URI = ''
MLFLOW_TRACKING_URI = ''
