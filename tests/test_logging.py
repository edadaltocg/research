import logging
from research.utils.logging import setup_logger


def main():
    # Call the setup_logging function to initialize the logger
    logger = logging.getLogger(__file__)

    # Log messages with different severity levels
    logger.debug("This is a debug message")  # This won't appear because the level is set to INFO
    logger.info("This is an info message")  # This will appear
    logger.warning("This is a warning message")  # This will also appear
    logger.error("This is an error message")  # This will also appear
    logger.critical("This is a critical message")  # This will also appear


if __name__ == "__main__":
    setup_logger(level=logging.DEBUG)
    main()
