from importlib.metadata import PackageNotFoundError, version

from .csv_logger import CSVLogger
from .wandb_logger import WANDBLogger
from .my_ex import MyEx


try:
    __version__ = version("my-exes")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CSVLogger",
    "MyEx",
    "WANDBLogger",
]
