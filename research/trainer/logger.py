from dataclasses import dataclass, field
import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from research.logger import CSVLogger


@dataclass
class TrainerLogger:
    log_dir: Path
    writer: SummaryWriter = field(init=False)
    logger: logging.Logger = field(init=False)
    csv_logger: CSVLogger = field(init=False)

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.writer = SummaryWriter(self.log_dir / "runs")
        self.csv_logger = CSVLogger(self.log_dir / "logs.csv")

    def log(self, metrics: dict[str, list[float]], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v[-1], step)
        self.csv_logger.log(*metrics.values())
