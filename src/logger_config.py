import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_module_logger(module_name: str):
    """
    Creates a dedicated logger for a module.
    Every module gets its own separate log file.
    """
    logger = logging.getLogger(module_name)

    # Prevent duplicate handlers if imported twice
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_path = os.path.join(LOG_DIR, f"{module_name}.log")

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
