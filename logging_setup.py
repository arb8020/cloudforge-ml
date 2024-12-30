# logging_setup.py

import logging
import logging.config
import yaml
from pathlib import Path
from json_formatter import JSONFormatter

def setup_logging(log_file_path: str, config_path='log_config.yaml', default_level=logging.INFO):
    """
    Sets up logging configuration.
    Updates the log file path in the configuration before applying it.
    """
    if Path(config_path).is_file():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f.read())

        # Update the 'filename' for the 'file' handler to the desired path
        config['handlers']['file']['filename'] = log_file_path

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_logger(name: str = 'my_logger') -> logging.Logger:
    """
    Returns a logger instance.
    """
    return logging.getLogger(name)
