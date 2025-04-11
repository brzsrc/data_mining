from datetime import timedelta

import pandas as pd

from assignment.data_loader import DataLoader
from assignment.task_1b import Task1B


class Task1C:
    """
    Essentially there are two approaches you can consider to create a predictive model
    using this dataset (which we will do in the next part of this assignment):

    -> use a machine learning approach that can deal with temporal data (e.g. recurrent neural networks)

    or you can try to aggregate the history somehow to create attributes that can be used in a more common machine
    learning approach (e.g. SVM, decision tree). For instance, you use the average mood during the last five
    days as a predictor. Ample literature is present in the area of temporal data mining that describes how
    such a transformation can be made.

    For the feature engineering, you are going to focus on such a transformation in this part of the assignment.
    This is illustrated in Figure 1.
    """
    df = Task1B.remove_incorrect_values()
    @classmethod
    def get_y(cls) -> pd.DataFrame:
        y = cls.df[cls.df.variable == "mood"].groupby(["id", "date"]).value.mean().reset_index()
        y["date"] += timedelta(days=1)
        return y.set_index(["id", "date"]).value.rename("y")

    @classmethod
    def aggregate_data_1day(cls):
        df = cls.df
        print(df.variable.unique())

        sum_cols = [
            "appCat.builtin",
            "appCat.communication",
            "appCat.entertainment",
            "appCat.finance",
            "appCat.game",
            "appCat.office",
            "appCat.other",
            "appCat.social",
            "appCat.travel",
            "appCat.unknown",
            "appCat.utilities",
            "appCat.weather",
            "call",
            "sms",
            "screen"
        ]
        group = df.groupby(["id", "date", "variable"]).value
        sum_agg = group.sum().unstack()[sum_cols]
        mean_agg = group.mean().unstack().drop(columns=sum_cols)
        assert set(df.variable.unique()) == set(sum_agg.columns) | set(mean_agg.columns)
        agg = sum_agg.join(mean_agg)
        print(f"dropping  {agg.mood.isna().sum()} / {len(agg)} rows")
        return agg[agg.mood.notna()].fillna(0)


# if __name__ == '__main__':
#     X = Task1C.aggregate_data_1day()
#     y = Task1C.get_y()
#     Xy = X.join(y)
