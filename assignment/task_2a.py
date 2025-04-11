from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from assignment.task_1c import Task1C


class Task2A:

    @classmethod
    def get_train_test(cls):
        x = Task1C.aggregate_data_1day()
        y = Task1C.get_y()
        y = y.mul(2).round().div(2).astype(str)
        xy = x.join(y).reset_index(drop=True).dropna()
        return train_test_split(xy[x.columns], xy['y'], test_size=0.2, shuffle=True)

    @classmethod
    def classification(cls):
        x_train, x_test, y_train, y_test = cls.get_train_test()
        print(x_train.shape, y_train.shape)
        # y = (y - y.mean()) / y.std()
        # y = y.round(1).astype(str)
        # y = y.mul(4).round().div(4).astype(str)
        # y = Task1C.get_y().round(1).astype(str)

        rf = RandomForestClassifier(max_depth=8)
        # rf = LogisticRegression()
        # Use random search to find the best hyperparameters
        # rf = RandomizedSearchCV(
        #     RandomForestClassifier(),
        #     param_distributions={'n_estimators': randint(50, 500),'max_depth': randint(20, 100)},
        #     n_iter=5,
        #     cv=5
        # )
        rf.fit(x_train, y_train)
        fig, axs = plt.subplots(2, 1)
        for name, x, y, ax in zip("train test".split(), (x_train, x_test), (y_train, y_test), axs.flatten()):
            y_pred = rf.predict(x)
            print(f"{name} accuracy: {(y_pred == y).mean()}")
            ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax).plot()
            ax.tick_params(axis='x', labelrotation=45)
        # ConfusionMatrixDisplay(cm, labels).plot()
        # plt.xticks(rotation=45)
        plt.show()

if __name__ == '__main__':
    Task2A.classification()
