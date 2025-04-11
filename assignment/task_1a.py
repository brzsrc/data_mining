from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from numpy.ma.core import negative, around

from assignment.data_loader import DataLoader
import matplotlib.pyplot as plt

class Task1A:
    """
    1.
    Notice all sorts of properties of the dataset: how many records are there, how many
    attributes, what kinds of attributes are there, ranges of values, distribution of values,
    relationships between attributes, missing values, and so on. A table is often a suitable
    way of showing such properties of a dataset. Notice if something is interesting (to you,
    or in general), make sure you write it down if you find something worth mentioning.

    2.
    Make various plots of the data. Is there something interesting worth reporting?
    Report the figures, discuss what is in them. What meaning do those bars, lines, dots, etc.
    convey? Please select essential and interesting plots for discussion, as you have limited
    space for reporting your findings.

    1.
    --
    how many records are there,
    how many attributes, what kinds of attributes are there,
    ranges of values
    --> shown by func text() //Done:
        after inspecting values, we can see there are uncorrected values since the time spent cannot be negative,
        so before we plot the graph about time spent on appCat, we first remove incorrect values (values < 0)

    distribution of values
    --> the distribution of values per variable

    relationships between attributes
    --> correlation matrix //TO DO

    missing values
    --> show missing values & deal with missing values: 1b

    2.
    ---
    interesting plots:
    plot1:
    --> n_records per person per day || the total number of records per person colored by date
    --> the total number of records per day ?
    --> the average number of records over person per day ?

    plot2:
    --> the n_records per variable per person ?
    --> the n_records per variable per day ?
    --> the average number of records over people per variable per day ?

    plot3:
    --> total time spent in appCat per person per day
    --> n_records per appCat variable where the recorded values > 3hr or < 1s
    -->



    Deliverables:
        -> n records
        -> n attributes
        -> kinds of attributes
        -> ranges of values
        -> distribution of values
        -> relationships between attributes
        -> missing values
        -> make various plots of the data

    """

    df = DataLoader.load_to_df()

    @classmethod
    def text(cls):
        """
        -> n records
        -> n attributes
        -> kinds of attributes
        -> ranges of values
        """
        df = cls.df.copy()

        print(f"n records: {len(df)}")
        print("attributes: [id (string), time (datetime), variable (string), value (float)]")
        n_users = len(df.id.unique())
        print(f"we have {n_users} user IDs")
        dates = sorted(df.date.unique())
        min_date, max_date = dates[0], dates[-1]
        print(f"the data spans over {len(dates)} days, from {min_date} to {max_date}")
        print("variables:")
        print(df.groupby("variable").value.agg(["count", "min", "max", "mean", "median"]).round(3).to_string())

        # print(df.groupby("variable").value.describe().to_string())


    @classmethod
    def variables_distribution_of_values(cls):
        print(len(cls.df.variable.unique()))
        fig, axes = plt.subplots(len(cls.df.variable.unique()) // 4 + 1, 4, figsize=(20, 12))
        axfl = axes.flatten()

        for (group, sdf), ax in zip(cls.df.groupby("variable"), axfl):
            ax.hist(sdf.value, bins=30, alpha=0.7, color='b')  # Histogram (normalized)
            # ax.plot
            ax.set_title(group)
            ax.grid(True)
        plt.suptitle("Distribution of values of each variable")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(len(cls.df.id.unique()) // 3, 3, figsize=(20, 20))
        axfl = axes.flatten()
        #
        # for (group, sdf), ax in zip(cls.df.groupby("variable"), axfl):
        #     # ax.hist(sdf.value, bins=30, alpha=0.7, color='b')  # Histogram (normalized)
        #     record_count = sdf.groupby(["id", "date"]).apply(len).unstack(0).fillna(0)
        #     print(record_count)
        #     ax.plot(record_count.index, record_count, label=group)
        #     ax.set_title(group)
        #     ax.grid(True)
        record_count = cls.df.groupby(["id", "date", "variable"]).apply(len).unstack().fillna(0)
        record_count_mood = record_count["mood"].unstack(0).fillna(0)
        for i, uid in enumerate(record_count_mood.columns):
            ax=axfl[i]
            ax.plot(record_count_mood.index, record_count_mood[uid], label=uid)
            ax.set_title(uid)
            ax.grid(True)
            # Rotate x-axis labels
            for label in ax.get_xticklabels():
                label.set_rotation(45)
            # truncated_labels = [label.strftime('%m-%d') for label in record_count_mood.index]
            # print(truncated_labels)
            # ax.set_xticklabels(truncated_labels, rotation=45, ha="right")
        plt.suptitle("Distribution of values of mood over dates")
        plt.tight_layout()
        plt.show()

        # for uid in record_count_mood.columns:
        #     plt.plot(record_count_mood.index, record_count_mood[uid], label=uid)
        #     plt.suptitle("Distribution of values of mood over dates")
        #     plt.tight_layout()
        #     plt.show()


    @classmethod
    def count_extreme_values_in_each_appCat(cls):
        df = cls.df[cls.df.variable.str.startswith('appCat')][["variable", "value"]]
        total_cnt = df.groupby("variable")["value"].count().to_frame()
        total_cnt.rename(columns={"value": "total_cnt"}, inplace=True)

        negative_values = df[df["value"] < 0].groupby(["variable"]).count()
        small_values = df[df["value"] < 1].groupby(["variable"]).count()
        large_values = df[df["value"] > 10800].groupby(["variable"]).count()

        result = pd.concat([negative_values, small_values, large_values], axis=1).fillna(0)
        result.columns = ['n_neg_vals', 'n_small_vals', 'n_large_vals']

        res_percentage = pd.merge(total_cnt, result, on='variable', how='right')
        # Calculate the percentage of positive values over total values
        res_percentage['neg_percentage'] = ((res_percentage['n_neg_vals'] / res_percentage['total_cnt']) * 100)
        res_percentage['small_percentage'] = ((res_percentage['n_small_vals'] / res_percentage['total_cnt'])* 100)
        res_percentage['large_percentage'] = ((res_percentage['n_large_vals'] / res_percentage['total_cnt'])* 100)
        print(res_percentage.to_string())

        fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
        # Hide axes
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=res_percentage.values,
                         colLabels=res_percentage.columns,
                         cellLoc='center',
                         loc='center')

        # Set table font size (optional)
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Adjust layout and show the plot
        plt.title('Summary Table of Value Counts by Date')
        plt.show()

    @classmethod
    def plot1(cls):
        """
        plot 1:
            - n records per person per day and total N records bar:
                shows that there was leading and trailing dates where very little was recorded, shows that there
                was a lot of noise per user on recording levels, and that no user


        """

        df = cls.df.groupby(["id", "date"]).apply(len).unstack(0).fillna(0)

        plt.plot(df.rolling(5).mean(), label=df.columns)
        plt.legend()
        plt.title("n_records per person per day")
        plt.show()

        # Plotting each day's values
        # plt.bar([c.removeprefix("AS14.") for c in df.columns], df.sum(), label=date)
        # plt.legend(title='Dates')
        # plt.title("the total number of records per person")
        # plt.show()

        # Reshape Data
        df = df.stack().unstack(0)
        num_colors = len(df.columns)
        colors = plt.cm.get_cmap('tab20', num_colors)  # Using a colormap to generate distinct colors
        color_list = [colors(i) for i in range(num_colors)]
        # Create a stacked bar plot
        df.plot(kind='bar', stacked=True, color=color_list, ax=plt.gca(), figsize=(15, 20))#legend=False
        plt.tight_layout()
        plt.title("the number of records per person colored by days")
        plt.show()

        # fig, axes = plt.subplots(5, 4, figsize=(12, 10))
        # for (group, sdf), ax in zip(cls.df.groupby("variable"), axes.flatten()):
        #     df = sdf.groupby(["id", "date"]).apply(len).unstack(0).fillna(0)
        #     ax.plot(df)
        #     ax.set_title(group)
        #     ax.imshow(df, aspect=.5)
        #     ax.set_yticks(list(range(len(df.index)))[::14])
        #     ax.set_xticks(list(range(len(df.columns)))[::4])
        #     ax.set_xticklabels(df.columns[::4], rotation=45, ha="right")
        #     ax.set_yticklabels(df.index[::14])

    @classmethod
    def plot2(cls):
        ...

    @classmethod
    def plot3(cls):
        """
        As we can see there are uncorrected values since the time spent cannot be negative,
        so before we plot the graph, we first remove incorrect values
        """
        df = cls.df[cls.df.variable.str.startswith('appCat')]
        df = df[["id", "date", "value"]][df["value"] > 0]
        df = df.groupby(["id", "date"]).value.sum().apply(lambda x: x / 3600).unstack(0).fillna(0)
        # Plot each user
        for uid in df.columns:
            plt.plot(df.index, df[uid], label=uid)
        # Adding labels and title
        plt.xlabel('Date')
        plt.ylabel('Time (hrs)')
        plt.title('Time spent on appCat per User Over Dates')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.legend(title='Users')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    @classmethod
    def _y(cls):
        sdf = cls.df[cls.df.variable == "mood"]
        df: pd.DataFrame = sdf.groupby(["id", "date"]).value.mean().reset_index(-1)
        df["date"] -= timedelta(days=1)
        y = df.stack("date")
        return y

    @classmethod
    def data_relationships(cls):
        """
        You need to think about whether this can be shown in 1c or not - it is much easier to find relationships
        on the cleaned data using machine learning techniques than using basic plots / manually combing through
        data.

        The most basic is just df.cov() on the cleaned + aggregated data.
        """
        ...

        df = cls.df.groupby(["id", "date", "variable"]).apply(len).unstack().fillna(0)
        # print(df[df["circumplex.arousal"] != 0].to_string())
        # print(len((df[df["circumplex.arousal"] != 0 & df["circumplex.valence"] != 0].index)) / len((df[df["mood"] != 0].index)))
        print(len(df[(df["circumplex.arousal"] != 0) & (df["circumplex.valence"] != 0) & (df["mood"] != 0)].index) / len((df[df["mood"] != 0].index)))
        print(
            len(df[(df["activity"] != 0) & (df["mood"] != 0)].index) / len(
                (df[df["mood"] != 0].index)))
        print(
            len(df[(df["screen"] != 0) & (df["mood"] != 0)].index) / len(
                (df[df["mood"] != 0].index)))

        # df = cls.df.groupby(["id", "datetime", "variable"]).apply(len).unstack()
        cnt = 0
        for group, sdf in list(cls.df.groupby(["id", "datetime", "variable"])):
            if len(sdf) > 1:
                cnt += 1
                print(sdf.to_string())
        print(cnt)

        # print(
        #     len(df[(df["screen"] != 0) & (df[df.index.startswith('appCat')] != 0)]) / len(
        #         (df[df["screen"] != 0].index)))

    @classmethod
    def main(cls):
        # cls.text()
        # cls.variables_distribution_of_values()
        # cls.plot1()
        cls.data_relationships()
        # cls.plot3()
        # cls.count_extreme_values_in_each_appCat()

    @classmethod
    def time_plot(cls):
        df = cls.df.copy()
        df = df[df.variable.str.startswith('appCat')]
        df = df[df.id == cls.df.id.unique()[0]]
        to_minute = lambda x: (t:=x.time()).hour * 100 + t.minute
        df["end"] = (df.datetime + df.value.apply(lambda x: timedelta(seconds=x))).apply(to_minute)
        df["start"] = df.datetime.apply(to_minute)
        for date, sdf in df.groupby("date"):
            sdfx = sdf.sort_values("start")
            plt.plot([x for s, e in zip(sdfx.start, sdfx.end) for x in [s, e]])
            plt.title(date)
            plt.show()


if __name__ == '__main__':
    # Task1A.time_plot()
    # Task1A.variables_distribution_of_values()
    Task1A.main()