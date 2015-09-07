import os

dir_src = os.path.dirname(os.path.abspath(__file__))
print dir_src
dir_main = os.path.split(dir_src)[0]
print dir_main
dir_log = os.path.join(dir_main, 'logs')

from optics import *

try:
    os.makedirs(dir_log)
except OSError as e:
    pass

import logger
logger.start_logger(os.path.join(dir_log, 'work.log'))

