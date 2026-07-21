import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Load CSV dataset"""
        self.df = pd.read_csv(self.filepath)
        return self.df

    def clean_data(self):
        """Basic cleaning"""
        self.df.drop_duplicates(inplace=True)

        # Remove completely empty rows
        self.df.dropna(how="all", inplace=True)

        # Fill numeric missing values
        numeric_cols = self.df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        return self.df

    def convert_date(self, column="Date"):
        """Convert Date column to datetime"""
        if column in self.df.columns:
            self.df[column] = pd.to_datetime(
                self.df[column],
                errors="coerce"
            )

            self.df = self.df.sort_values(column)

        return self.df

    def prepare_timeseries(self, date_col="Date", target_col="Sales"):
        """Prepare data for forecasting"""

        df = self.df[[date_col, target_col]].copy()

        df[date_col] = pd.to_datetime(df[date_col])

        df = df.set_index(date_col)

        df = df.asfreq("D")

        df[target_col] = df[target_col].interpolate()

        return df

    def save_clean_data(self, output_path):
        self.df.to_csv(output_path, index=False)


if __name__ == "__main__":

    processor = DataPreprocessor("../data/train.csv")

    processor.load_data()

    processor.clean_data()

    processor.convert_date()

    processor.save_clean_data("../data/clean_train.csv")

    print("Data preprocessing completed successfully.")
