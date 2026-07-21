import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


class ModelEvaluator:

    def __init__(self, actual, predicted):

        self.actual = np.array(actual)
        self.predicted = np.array(predicted)

    def mae(self):
        return mean_absolute_error(
            self.actual,
            self.predicted
        )

    def rmse(self):
        return np.sqrt(
            mean_squared_error(
                self.actual,
                self.predicted
            )
        )

    def mape(self):
        return (
            mean_absolute_percentage_error(
                self.actual,
                self.predicted
            ) * 100
        )

    def evaluate(self):

        print("=" * 40)
        print("MODEL PERFORMANCE")
        print("=" * 40)

        print(f"MAE  : {self.mae():.2f}")
        print(f"RMSE : {self.rmse():.2f}")
        print(f"MAPE : {self.mape():.2f}%")

        print("=" * 40)


if __name__ == "__main__":

    actual = [120,130,150,160,170,180]

    predicted = [118,132,148,162,169,183]

    evaluator = ModelEvaluator(
        actual,
        predicted
    )

    evaluator.evaluate()
