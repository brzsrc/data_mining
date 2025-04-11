import datetime as dt
from pathlib import Path

import pandas as pd

class DataLoader:
    CSV_FILE = Path(__file__).parent / 'static/data.csv'

    @classmethod
    def load_to_df(cls) -> pd.DataFrame:
        df = pd.read_csv(cls.CSV_FILE).drop(columns="Unnamed: 0")
        df["datetime"] = df["time"].apply(dt.datetime.fromisoformat)
        df["date"] = df["datetime"].dt.date
        df["time"] = df["datetime"].dt.time
        df = df[["id", "datetime", "date", "time", "variable", "value"]]
        df["value"] = df["value"].astype(float)
        return df

