import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


def get_logger(base_dir: Path | None = None) -> "Logger":
    base_dir = base_dir if base_dir else Path(".")
    log_dir: Path = Path(base_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # Console
    logger.add(sys.stdout, level="DEBUG", backtrace=True, diagnose=True, enqueue=True)

    # DEBUG / INFO / WARNING  ->  stdout.log
    logger.add(
        log_dir / "stdout.log",
        level="DEBUG",
        filter=lambda r: r["level"].no < logger.level("ERROR").no,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
    )

    # ERROR / CRITICAL  ->  stderr.log
    logger.add(
        log_dir / "stderr.log",
        level="ERROR",
        rotation="10 MB",
        retention="20 days",
        compression="zip",
        enqueue=True,
    )

    # All levels, structured JSON  ->  structured.log.json
    logger.add(
        log_dir / "structured.log.json",
        level="DEBUG",
        serialize=True,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
    )

    return logger
