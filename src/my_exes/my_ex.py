from typing import Any
from pathlib import Path
from datetime import datetime as dt

from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import Module

from .csv_logger import CSVLogger


class MyEx:
    def __init__(self, config_file: str | Path) -> None:
        self.config_file = Path(config_file).resolve()
        self.cfg = OmegaConf.load(self.config_file)

        if "name" not in self.cfg:
            self.cfg["name"] = "experiment"  # type: ignore

        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        if "log_dir" not in self.cfg:
            self.cfg["log_dir"] = self.cfg.name + timestamp  # type: ignore
        else:
            self.log_dir = self.cfg["log_dir"] + timestamp  # type: ignore
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.loggers = self.cfg.get("loggers", ["tensorboard"])  # type: ignore
        self.tb_logger = SummaryWriter(logdir=str(self.log_dir))
        self.csv_logger = None
        if "csv" in self.loggers:
            self.csv_logger = CSVLogger(self.log_dir / "log.csv")

    def log(
        self, state: str, category: str, value: Any, iteration: int, note: str = ""
    ) -> None:
        if isinstance(value, (int, float)):
            self.tb_logger.add_scalar(
                f"{category}/{state}", value, iteration, summary_description=note
            )

        if self.csv_logger is not None:
            self.csv_logger.log(state, category, value, iteration, note)

    def log_model_graph(self, model: Module, sample_input: Tensor) -> None:
        self.tb_logger.add_graph(model, input_to_model=sample_input)
