import logging
import os
import sys

# format of the logging message
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# set the config
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# create an object of the logger
logger = logging.getLogger("AUTOFORECASTLogger")
