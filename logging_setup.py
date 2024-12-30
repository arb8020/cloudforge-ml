# logging_setup.py

import logging
import logging.config
import yaml
from pathlib import Path
from json_formatter import JSONFormatter

def setup_logging(log_file_path: str = None, config_path='log_config.yaml', default_level=logging.INFO):
    """
    Sets up logging configuration.
    If log_file_path is provided, logs will be written to the specified file.
    Otherwise, logs will only be output to stdout and stderr.

    Args:
        log_file_path (str, optional): Path to the log file. Defaults to None.
        config_path (str): Path to the logging configuration YAML file.
        default_level (int): Default logging level.
    """
    if Path(config_path).is_file():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f.read())

        if log_file_path:
            # Update the 'filename' for the 'file' handler to the desired path
            if 'file' in config['handlers']:
                config['handlers']['file']['filename'] = log_file_path
            else:
                # If 'file' handler is not defined, you might want to handle it differently
                pass
        else:
            # Remove file handlers if log_file_path is not provided
            handlers_to_remove = []
            for handler_name, handler in config['handlers'].items():
                if handler_name == 'file':
                    handlers_to_remove.append(handler_name)
            for handler_name in handlers_to_remove:
                del config['handlers'][handler_name]
            # Remove 'file' from loggers
            for logger_name, logger in config['loggers'].items():
                if 'file' in logger['handlers']:
                    logger['handlers'].remove('file')
            if 'file' in config['root']['handlers']:
                config['root']['handlers'].remove('file')

        logging.config.dictConfig(config)
    else:
        # Basic configuration: logs to stdout and stderr
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.StreamHandler(sys.stderr)
            ]
        )

def get_logger(name: str = 'my_logger') -> logging.Logger:
    """
    Returns a logger instance.

    Args:
        name (str, optional): Name of the logger. Defaults to 'my_logger'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
