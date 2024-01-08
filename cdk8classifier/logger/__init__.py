# https://data-flair.training/blogs/python-logging/
# https://tutorialedge.net/python/python-logging-best-practices/
# https://www.blog.pythonlibrary.org/2014/02/11/python-how-to-create-rotating-logs/
# https://stackoverflow.com/a/13733863

import logging
import sys
from datetime import date
from logging.handlers import TimedRotatingFileHandler
from pathlib import PurePosixPath

from cdk8classifier.commons import configs, utils

logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("hyperopt.tpe").setLevel(logging.ERROR)
logging.getLogger("hyperopt.fmin").setLevel(logging.ERROR)
logging.getLogger("hyperopt.pyll.base").setLevel(logging.ERROR)

LEVEL = configs['project_log_level']

if LEVEL == 'ERROR':
    _level = logging.ERROR
elif LEVEL == 'INFO':
    _level = logging.INFO
else:
    _level = logging.DEBUG

filename = PurePosixPath(utils.get_project_root()) / 'logs' / f'classifier-{date.today()}.log'
# logging.basicConfig(filename=filename, level=_level, format='%(asctime)s [%(levelname)s] %(message)s')

_logger = logging.getLogger('classifier_')
_logger.setLevel(_level)
_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

file_handler = TimedRotatingFileHandler(filename, when='midnight')
file_handler.setFormatter(_formatter)
_logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(_formatter)
_logger.addHandler(console_handler)


def log_error(**kwargs):
    _logger.error(f"[classifier] {kwargs['msg']}")


def log_trace(**kwargs):
    _logger.debug(f"[classifier] {kwargs['msg']}")


def log_info(**kwargs):
    _logger.info(f"[classifier] {kwargs['msg']}")
