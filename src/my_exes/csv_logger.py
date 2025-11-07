from pathlib import Path

import pandas as pd


class CSVLogger:
    def __init__(self, log_file: str | Path = "log.csv"):
        self.log_file = log_file
        # state: train, test, validation, ...
        # category: loss, accuracy, ...
        columns = [
            "timestamp", "state", "category",
            "value", "iteration", "note"
        ]
        self.df = pd.DataFrame(columns=columns)

    def log(
        self, state: str, category: str,
        value: int | float, iteration: int, note=""
    ):
        row = {
            "timestamp": pd.Timestamp.now(),
            "state": state,
            "category": category,
            "value": value,
            "iteration": iteration,
            "note": note
        }
        new_row_df = pd.DataFrame([row])
        if not new_row_df.empty:
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        self.save()

    def save(self) -> None:
        self.df.to_csv(self.log_file, index=False)
