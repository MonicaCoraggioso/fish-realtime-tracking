"""
Logging configuration for fish tracking system.

Provides centralized logging setup with appropriate handlers and formatters
for console and file output.

Author: Refactored from original code by Monica Coraggioso
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    console_output=True,
    format_string=None
):
    """Configure logging for the tracking system.

    Parameters
    ----------
    log_level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str or Path, optional
        Path to log file. If None, no file logging is performed.
    console_output : bool, default=True
        If True, also log to console (stdout)
    format_string : str, optional
        Custom format string. If None, uses default format.

    Returns
    -------
    logger : logging.Logger
        Configured root logger

    Examples
    --------
    # Basic setup with console output only
    logger = setup_logging()

    # Setup with file logging
    logger = setup_logging(
        log_level=logging.DEBUG,
        log_file='tracking_2024-01-15.log'
    )

    # Use in code
    logger.info("Starting tracking...")
    logger.warning("Fish lost at frame 123")
    logger.error("Registration failed", exc_info=True)
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Define format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name):
    """Get a logger with the specified name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the module)

    Returns
    -------
    logger : logging.Logger
        Logger instance

    Examples
    --------
    # In a module
    logger = get_logger(__name__)
    logger.info("Processing frame...")
    """
    return logging.getLogger(name)


def create_experiment_logger(experiment_name, output_dir='logs'):
    """Create a logger for a specific experiment with automatic file naming.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    output_dir : str or Path, default='logs'
        Directory where log files will be stored

    Returns
    -------
    logger : logging.Logger
        Configured logger for this experiment

    Examples
    --------
    logger = create_experiment_logger('viscosity_run_01')
    logger.info("Starting experiment...")
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = Path(output_dir) / log_filename

    return setup_logging(
        log_level=logging.INFO,
        log_file=log_path,
        console_output=True
    )


def log_function_call(func):
    """Decorator to log function entry and exit.

    Parameters
    ----------
    func : callable
        Function to wrap

    Returns
    -------
    wrapper : callable
        Wrapped function that logs calls

    Examples
    --------
    @log_function_call
    def process_frame(frame_number):
        # Processing code
        pass
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}", exc_info=True)
            raise

    return wrapper


# Default logger instance
_default_logger = None


def get_default_logger():
    """Get the default logger, initializing it if necessary.

    Returns
    -------
    logger : logging.Logger
        Default logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
