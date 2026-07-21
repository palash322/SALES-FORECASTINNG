import pandas as pd
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class ProphetForecaster:

    def __init__(self, filepath):
        self.filepath = filepath
        self.model = Prophet()
        self.df = None

    def load_data(self):

        self.df = pd.read_csv(self.filepath)

        self.df = self.df.rename(
            columns={
                "Date": "ds",
                "Sales": "y"
            }
        )

        self.df["ds"] = pd.to_datetime(self.df["ds"])

    def train(self):

        self.model.fit(self.df)

        joblib.dump(
            self.model,
            "models/prophet.pkl"
        )

    def forecast(self, days=30):

        future = self.model.make_future_dataframe(
            periods=days
        )

        forecast = self.model.predict(future)

        return forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]

    def evaluate(self):

        prediction = self.model.predict(self.df)

        mae = mean_absolute_error(
            self.df["y"],
            prediction["yhat"]
        )

        rmse = np.sqrt(
            mean_squared_error(
                self.df["y"],
                prediction["yhat"]
            )
        )

        print("MAE :", mae)
        print("RMSE:", rmse)


if __name__ == "__main__":

    model = ProphetForecaster(
        "data/train.csv"
    )

    model.load_data()

    model.train()

    model.evaluate()

    print(model.forecast(30).tail())
