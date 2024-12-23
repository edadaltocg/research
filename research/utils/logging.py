import logging
import torch.distributed as dist
import sys
from typing import Union

# Updated LOG_FORMAT with fill characters (.)
LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI escape codes for different colors
COLOR_CODES = {
    "DEBUG": "\033[0;37m",  # White
    "INFO": "\033[0;32m",  # Green
    "WARNING": "\033[0;33m",  # Yellow
    "ERROR": "\033[0;31m",  # Red
    "CRITICAL": "\033[1;31m",  # Bold Red
    "DATE": "\033[0;33m",  # Yellow for the date
    "FUNC_NAME": "\033[0;34m",  # Blue for the function name
    "RESET": "\033[0m",  # Reset to normal
}


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        # Colorize the levelname
        levelname_color = COLOR_CODES.get(record.levelname, COLOR_CODES["RESET"])
        record.levelname = f"{levelname_color}{record.levelname}{COLOR_CODES['RESET']}"

        # Colorize the date
        date_str = self.formatTime(record, self.datefmt)
        record.asctime = f"{COLOR_CODES['DATE']}{date_str}{COLOR_CODES['RESET']}"

        # Colorize the function name
        func_name = record.funcName
        record.funcName = f"{COLOR_CODES['FUNC_NAME']}{func_name}{COLOR_CODES['RESET']}"

        # Finally, call the parent class format function
        return super().format(record)


def setup_logger(
    level: Union[int, str] = logging.INFO,
    rank_filter=True,
):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = ColorFormatter(LOG_FORMAT, DATE_FORMAT)
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = [handler]  # Override any existing handlers
    if rank_filter:
        logger.addFilter(RankFilter())
    return logger


class CSVLogger:
    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, "+a") as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=",", file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv, strict=False), 1):
                end = "," if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class RankFilter(logging.Filter):
    def __init__(self, func=None):
        """default filter is to log only if rank == 0, but users can write their lamda function for filtering"""
        super().__init__()
        self.func = func if func else lambda rank: rank == 0
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

    def filter(self, record):
        if self.func(self.rank):
            return True
        else:
            return False
