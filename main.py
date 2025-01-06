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

# Preserve uploaded data in session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Store uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.uploaded_data = pd.read_csv(uploaded_file)  # Read CSV immediately
    st.sidebar.success("File uploaded successfully!")

# Analyze button
if st.sidebar.button("Analyze Data"):
    # Use uploaded data if available
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
    else:
        with st.spinner("Loading data..."):
            data = load_data(selected_stock, start_date, end_date)

    # Check if required columns exist
    if "Date" not in data.columns or "Close" not in data.columns:
        st.error("Uploaded data must contain 'Date' and 'Close' columns.")
    else:
        # Forecasting
        df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        df_train = df_train.dropna()
        df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
        df_train = df_train.dropna()

        if df_train.empty:
            st.error("No valid data available after cleaning. Please check the input data.")
        else:
            # Prophet Model
            model = Prophet()
            model.fit(df_train)
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)

            # Display forecast
            st.write("Forecast Results:", forecast.head())
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig)

            # Optional: Plot original data
            plot_data(data)
