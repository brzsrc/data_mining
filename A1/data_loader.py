from collections import defaultdict
from pathlib import Path
from typing import List, NamedTuple, Dict

import csv
import numpy as np
import pandas as pd


class UserData:
    def __init__(self, id:str):
        self.id = id
        self.time = []
        #{time: {mood: 1, call:3, xxx:0.2334}}
        self.time_tbl = defaultdict(lambda: defaultdict(list[float]))

    id: str
    time: List[str]
    time_tbl: defaultdict[lambda: defaultdict[List[float]]]



class DataLoader:
    user_datas: Dict[str, UserData]

    def __init__(self):
        self.user_datas = {}

    class FILES:
        _FOLDER = Path(__file__).parent
        DEV = _FOLDER / "dataset_mood_smartphone.csv"

    def _get_user_data(self, id:str) -> UserData:
        if id not in self.user_datas.keys():
            self.user_datas[id] = UserData(id)
        return self.user_datas[id]

    def parse_dataset(self, file: Path) -> Dict[str, UserData]:
        assert file.suffix == ".csv", file.suffix

        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                id = row[1]
                time = row[2].split()[0]
                var = row[3]
                val = row[4]
                user_data = self._get_user_data(id)
                user_data.time.append(time)
                user_data.time_tbl[time][var].append(val)

        return self.user_datas


    def print_output(self):
        for user_data in self.user_datas.values():
            print(user_data.id)
            for time, time_tbl in user_data.time_tbl.items():
                print(time, time_tbl.items())


    @staticmethod
    def parse_dataset_2(file: Path) -> List[UserData]:
        assert file.suffix == ".csv", file.suffix
        user_data = pd.read_csv(file,  names=['id', 'time', 'variable', 'value'], dtype={'value': str})
        # print(user_data.head(5))
        user_data_grouped = user_data.groupby(["id", "time"])
        print(len(user_data_grouped.groups))
        cnt_group = 0
        max_group_len = max(len(group) for _, group in user_data_grouped)
        for i, (name, group) in enumerate(user_data_grouped):
            # if i == 100:
            #     break
            if len(group) == max_group_len:
                cnt_group += 1
                print(name)
                print(group)
        print(cnt_group)


if __name__ == '__main__':
    loader = DataLoader()
    loader.parse_dataset(DataLoader.FILES.DEV)
    loader.print_output()