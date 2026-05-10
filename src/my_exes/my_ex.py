from typing import Any
from pathlib import Path
from datetime import datetime as dt

import torch
from torch.nn import Module
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter
from torchvista import trace_model

from .csv_logger import CSVLogger
from .wandb_logger import WANDBLogger


class MyEx:
    def __init__(
        self,
        config_file: str | Path,
        exact_log_dir: str | Path | None = None,
    ) -> None:
        """Initialize MyEx experiment logger .

        Args:
            config_file: Path to the YAML configuration file for the experiment.
            exact_log_dir: Optional path to an existing directory for logging.
                Useful for test/prediction from the previously created logs.
                If None, a timestamped directory will be created based on
                the experiment name.

        Raises:
            AssertionError: If exact_log_dir is provided
            but does not exist or is not a directory.
        """
        # load configurations
        self.config_file = Path(config_file).resolve()
        self.cfg: DictConfig = OmegaConf.load(self.config_file)  # type: ignore

        # self.log_dir: Path = Path(".")
        self.loggers: list[str] = []
        self.csv: CSVLogger | None = None
        self.tboard: SummaryWriter | None = None
        self.wandb: WANDBLogger | None = None

        # default experiment name
        if "name" not in self.cfg:
            self.cfg["name"] = "experiment"

        # set the log directory
        if exact_log_dir is not None:
            # validate and use the provided log directory
            self.log_dir = Path(exact_log_dir).resolve()
            assert self.log_dir.exists() and self.log_dir.is_dir(), \
                "exact_log_dir should point to an existing directory."
        else:
            # create a new log directory with timestamp
            timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            if "log_dir" not in self.cfg:
                self.log_dir = Path(self.cfg.name + "_" + timestamp)
            else:
                self.log_dir = Path(self.cfg["log_dir"]).resolve()
                self.log_dir = self.log_dir.joinpath(self.cfg.name + "_" + timestamp)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # save the config file in the log dir
        if self.config_file.parent != self.log_dir:
            OmegaConf.save(self.cfg, self.log_dir / "config.yaml")

        self.loggers: list[str] = self.cfg.get("loggers", [])
        self._init_loggers()

    def _init_loggers(self) -> None:
        """Initialize loggers based on the configuration."""
        # add csv logger as default if not specified
        if not self.loggers:
            self.loggers.append("csv")
        self.csv = CSVLogger(self.log_dir / "log.csv")

        if "tensorboard" in self.loggers:
            self.tboard = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            self.tboard.add_text("config", OmegaConf.to_yaml(self.cfg), 0)

        if "wandb" in self.loggers:
            self.wandb = WANDBLogger(
                log_dir=self.log_dir,
                project=self.cfg.get("wandb_project", self.cfg.name),
                config=OmegaConf.to_container(self.cfg, resolve=True)  # type: ignore
            )

    def log(
        self,
        state: str,
        category: str,
        value: Any,
        iteration: int,
        note: str = ""
    ) -> None:
        """Log into experiment loggers.

        Args:
            state (str): state of the experiment, e.g., "train", "validation", "test".
            category (str): category of the metric, e.g., "loss", "accuracy".
            value (Any): value of the metric to log.
            iteration (int): iteration number for the log entry.
            note (str, optional): additional notes for the csv log entry. Defaults to "".
        """
        if self.tboard:
            if isinstance(value, (int, float)):
                self.tboard.add_scalar(
                    f"{category}/{state}",
                    value,
                    global_step=iteration,
                    summary_description=note
                )
            else:
                self.tboard.add_text(
                    f"{category}/{state}",
                    str(value),
                    global_step=iteration
                )

        if self.wandb:
            self.wandb.add_scalar(
                f"{category}/{state}",
                value,
            )

        if self.csv:
            self.csv.log(state, category, value, iteration, note)

    def log_model_graph(self, model: Module, sample_input: torch.Tensor) -> None:
        trace_model(
            model,
            inputs=sample_input,
            collapse_modules_after_depth=0,
            export_path=self.log_dir / "model.html",
            export_format="html"
        )
        if self.tboard:
            self.tboard.add_graph(model, input_to_model=sample_input)
        if self.wandb:
            self.wandb.log_html(self.log_dir / "model.html")

    def save_model(self, model: Module, name: str = "model.pth") -> Path:
        """Save the model checkpoint.

        Args:
            model (Module): PyTorch Module to be saved.
            name (str, optional): Name of the file to save the model to.
            Defaults to "model.pth".

        Returns:
            Path: Path to the saved model file.
        """
        torch.save(model.state_dict(), self.log_dir / name)
        return self.log_dir / name

    def log_scalars(
        self,
        state: str,
        scalars: dict[str, float | int],
        iteration: int
    ) -> None:
        """Log multiple scalar values to experiment loggers.

        Args:
            state (str): state of the experiment, e.g., "train", "validation", "test".
            scalars (dict[str, float | int]): dictionary of scalar values to log.
            iteration (int): iteration number for the log entry.
        """
        if self.tboard:
            self.tboard.add_scalars(state, scalars, iteration)
        if self.wandb:
            self.wandb.add_scalars(state, scalars)

    def save_config(self, config_file: Path | str | None = None) -> None:
        """Save current config to a new file or update the config yaml file.

        Args:
            config_file (Path | str | None, optional): if provided,
            saves it at that location otherwise updates the original config file.
            Defaults to None.
        """
        _config_file = config_file or self.config_file
        with open(_config_file, "w") as f:
            OmegaConf.save(self.cfg, f)
        # also update the copy inside log dir
        OmegaConf.save(self.cfg, self.log_dir / "config.yaml")

    def close(self) -> None:
        if self.tboard:
            self.tboard.close()
        if self.wandb:
            self.wandb.run.finish()
