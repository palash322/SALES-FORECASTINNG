import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Sales Forecasting System")
st.markdown("Forecast future sales using Machine Learning")

uploaded_file = st.file_uploader(
    "Upload Sales Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_columns) == 0:
        st.error("No numeric sales column found.")
        st.stop()

    sales_column = st.selectbox(
        "Select Sales Column",
        numeric_columns
    )

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        x = "Date"
    else:
        x = df.index

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{df[sales_column].sum():,.0f}")
    col2.metric("Average Sales", f"{df[sales_column].mean():.2f}")
    col3.metric("Maximum Sales", f"{df[sales_column].max():,.0f}")

    fig = px.line(
        df,
        x=x,
        y=sales_column,
        title="Sales Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast")

    days = st.slider(
        "Forecast Days",
        7,
        90,
        30
    )

    st.info(
        f"Forecast for next {days} days will appear here after training the model."
    )

else:
    st.info("Upload a CSV file to begin.")
