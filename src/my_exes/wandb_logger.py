from pathlib import Path

import wandb


class WANDBLogger:
    def __init__(
        self,
        log_dir: Path,
        project: str,
        config: dict | None = None,
        **kwargs
    ) -> None:
        """
        Initialize the WANDBLogger.

        Args:
            project (str): The name of the W&B project.
            log_dir (Path): The directory where logs will be stored.
            **kwargs: Additional arguments passed to wandb.init().
        """
        if not wandb.login():
            raise RuntimeError("W&B login failed! Please check your W&B credentials.")

        self.run = wandb.init(
            project=project,
            dir=log_dir / "wandb",
            config=config,
            **kwargs
        )

    def log(
        self,
        data: dict,
        step: int | None = None,
        commit: bool | None = None
    ):
        """
        Log data to W&B.

        Args:
            data (dict): A dictionary of data to log.
            step (int, optional): The global step value for logging.
            commit (bool | None, optional): Whether to commit the log entry.
        """
        self.run.log(data, step=step, commit=commit)

    def add_scalar(
        self,
        tag: str,
        scalar: int | float,
        step: int | None = None
    ):
        """
        Log a scalar value to W&B, compatible with tensorboardX SummaryWriter.add_scalar.

        Args:
            tag (str): The tag name for the scalar.
            value: The scalar value to log.
            global_step: The global step value.
        """
        values = {tag: scalar}
        if step is not None:
            values.update({"global_step": step})
        self.run.log(values)

    def add_scalars(
        self,
        state: str,
        scalars: dict,
        step: int | None = None
    ):
        """
        Log multiple scalar values to W&B.

        Args:
            tag (str): The tag name for the scalar.
            scalars (dict): A dictionary of tag names to scalar values.
            global_step: The global step value.
        """
        values = {f"{state}/{k}": v for k, v in scalars.items()}
        if step is not None:
            values.update({"global_step": step})
        self.run.log(values)

    def log_html(self, html_path: Path, step: int | None = None):
        """
        Log an HTML file to W&B.

        Args:
            html_path (Path): The path to the HTML file to log.
            step (int): The global step value for logging.
        """
        self.run.log({"html": wandb.Html(str(html_path))}, step=step)
