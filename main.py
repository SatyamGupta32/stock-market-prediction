from time import sleep
import uuid
import pandas as pd
from sklearn.metrics import mean_absolute_error
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf

# Set up page configuration
st.set_page_config(layout="wide", page_title="Forcastify", page_icon="📈")

# Sidebar
st.sidebar.markdown(
    "<h1 style='text-align: center; font-size: 30px;'><b>Forcasti.</b><b style='color: orange'>fy</b></h1>",
    unsafe_allow_html=True,
)
st.sidebar.title("Options")

start_date_key = str(uuid.uuid4())
start_date = st.sidebar.date_input("Start date", date(2018, 1, 1), key=start_date_key)
end_date = st.sidebar.date_input("End date", date.today())

stocks = (
    "AAPL",
    "GOOG",
    "MSFT",
    "GME",
    "AMC",
    "TSLA",
    "AMZN",
    "NFLX",
    "NVDA",
    "AMD",
    "PYPL",
)

selected_stock = st.sidebar.selectbox("Select stock for prediction", stocks)
selected_stocks = st.sidebar.multiselect("Select stocks for comparison", stocks)

years_to_predict = st.sidebar.slider("Years of prediction:", 1, 5)
period = years_to_predict * 365

# Main title
st.markdown(
    "<h1 style='text-align: center;'>Stock Forecast App 📈</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'><b>Forcasti.</b><b style='color: orange'>fy</b> is a simple web app for stock price prediction using the <a href='https://facebook.github.io/prophet/'>Prophet</a> library.</p>",
    unsafe_allow_html=True,
)

selected_tab = option_menu(
    menu_title=None,
    options=["Dataframes", "Plots", "Statistics", "Forecasting", "Comparison"],
    icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up"],
    menu_icon="📊",
    default_index=0,
    orientation="horizontal",
)

# Helper functions
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df

def plot_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_multiple_data(forecasts, stocks):
    fig = go.Figure()
    for i, stock in enumerate(stocks):
        fig.add_trace(go.Scatter(x=forecasts[i]["ds"], y=forecasts[i]["yhat"], name=stock))
    fig.layout.update(title_text="Forecast Comparison", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_volume(data):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name="Volume"))
    fig.layout.update(title_text="Stock Volume Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Data loading
with st.spinner("Loading data..."):
    data = load_data(selected_stock, start_date, end_date)
    sleep(1)
# Forecasting
df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
# Ensure 'y' is numeric and handle missing values
if "Close" not in df_train.columns or df_train.empty:
    st.error("No valid data available for forecasting. Please adjust the date range or select a different stock.")
else:
    df_train = df_train.dropna()  # Drop rows with missing values
    df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")  # Coerce invalid values to NaN
    df_train = df_train.dropna()  # Drop rows with NaN after conversion

    if df_train.empty:
        st.error("No valid data available after cleaning. Please select a different stock or date range.")
    else:
        model = Prophet()
        model.fit(df_train)
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

        end_date_datetime = pd.to_datetime(end_date)
        forecast = forecast[forecast["ds"] >= end_date_datetime]
