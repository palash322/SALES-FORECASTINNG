import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


class LSTMForecaster:

    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = MinMaxScaler()
        self.model = None

    def load_data(self):

        df = pd.read_csv(self.filepath)

        self.sales = df["Sales"].values.reshape(-1, 1)

        self.scaled = self.scaler.fit_transform(self.sales)

    def prepare_data(self, window=30):

        X = []
        y = []

        for i in range(window, len(self.scaled)):
            X.append(self.scaled[i-window:i])
            y.append(self.scaled[i])

        X = np.array(X)
        y = np.array(y)

        return X, y

    def build_model(self):

        self.model = Sequential()

        self.model.add(
            LSTM(
                64,
                input_shape=(30,1)
            )
        )

        self.model.add(Dense(32, activation="relu"))

        self.model.add(Dense(1))

        self.model.compile(
            optimizer="adam",
            loss="mse"
        )

    def train(self):

        X, y = self.prepare_data()

        self.build_model()

        stop = EarlyStopping(
            monitor="loss",
            patience=5
        )

        self.model.fit(
            X,
            y,
            epochs=20,
            batch_size=16,
            callbacks=[stop],
            verbose=1
        )

        self.model.save("models/lstm.keras")

    def predict_next(self):

        last = self.scaled[-30:]

        last = np.expand_dims(last, axis=0)

        pred = self.model.predict(last)

        return self.scaler.inverse_transform(pred)


if __name__ == "__main__":

    model = LSTMForecaster("data/train.csv")

    model.load_data()

    model.train()

    print(model.predict_next())
