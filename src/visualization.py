import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class SalesVisualizer:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])

    def sales_trend(self, sales_col="Sales"):

        fig = px.line(
            self.df,
            x="Date",
            y=sales_col,
            title="Sales Trend",
            markers=True
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white"
        )

        return fig

    def monthly_sales(self, sales_col="Sales"):

        monthly = self.df.copy()

        monthly["Month"] = monthly["Date"].dt.strftime("%b")

        summary = monthly.groupby("Month")[sales_col].sum().reset_index()

        fig = px.bar(
            summary,
            x="Month",
            y=sales_col,
            title="Monthly Sales"
        )

        return fig

    def forecast_vs_actual(
        self,
        actual_dates,
        actual_sales,
        forecast_dates,
        forecast_sales
    ):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=actual_dates,
                y=actual_sales,
                mode="lines",
                name="Actual"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_sales,
                mode="lines",
                name="Forecast"
            )
        )

        fig.update_layout(
            title="Forecast vs Actual",
            template="plotly_white"
        )

        return fig

    def correlation(self):

        numeric = self.df.select_dtypes(include="number")

        fig = px.imshow(
            numeric.corr(),
            text_auto=True,
            title="Correlation Matrix"
        )

        return fig


if __name__ == "__main__":

    df = pd.read_csv("data/train.csv")

    chart = SalesVisualizer(df)

    chart.sales_trend().show()
