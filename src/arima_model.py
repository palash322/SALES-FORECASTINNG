import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class ARIMAForecaster:

    def __init__(self, filepath):
        self.filepath = filepath
        self.model = None
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date")
        self.df = self.df.set_index("Date")

    def train(self):

        train = self.df["Sales"]

        self.model = ARIMA(
            train,
            order=(5,1,0)
        ).fit()

        joblib.dump(
            self.model,
            "models/arima.pkl"
        )

    def forecast(self, days=30):

        prediction = self.model.forecast(
            steps=days
        )

        return prediction

    def evaluate(self):

        actual = self.df["Sales"]

        pred = self.model.predict(
            start=1,
            end=len(actual)-1
        )

        mae = mean_absolute_error(
            actual[1:],
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual[1:],
                pred
            )
        )

        print("MAE :", mae)
        print("RMSE:", rmse)


if __name__ == "__main__":

    model = ARIMAForecaster(
        "data/train.csv"
    )

    model.load_data()

    model.train()

    model.evaluate()

    print(model.forecast(30))
