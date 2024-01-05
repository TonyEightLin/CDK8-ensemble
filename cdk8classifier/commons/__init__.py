from pathlib import PurePosixPath

from cdk8classifier.commons import utils
from cdk8classifier.commons.utils import MODEL_DATA_FILE_PATH

configs = utils.get_config_dic()

PROJECT_ROOT_PATH = utils.get_project_root()
TRAINING_ROOT_PATH = PurePosixPath(PROJECT_ROOT_PATH) / 'resources' / 'training'
PRODUCTION_ROOT_PATH = PurePosixPath(PROJECT_ROOT_PATH) / 'resources' / 'production'

FRAG_FILE_PATH = configs['datasource']['fragments']
TRAINING_CDK8_FILE_PATH = configs['datasource']['training']['cdk8']
PRODUCTION_CDK8_FILE_PATH = configs['datasource']['production']['cdk8']
