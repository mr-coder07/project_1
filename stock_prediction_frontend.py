import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import LSTMCell
import os
import pickle

# Set page configuration
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")

# Load custom CSS for styling
def load_css():
    css = """
    <style>
        body {
            background-color: #F0F8FF;
            color: #161D6F;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1, h2, h3 {
            color: #0B2F9F;
            animation: fadeInDown 1s ease;
        }
        .stButton>button {
            background-color: #0B2F9F;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            background-color: #161D6F;
            transform: scale(1.1);
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
        }
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1s ease;
        }
        th, td {
            border: 1px solid #0B2F9F;
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #0B2F9F;
            color: white;
        }
        td {
            background-color: #E6F2FF;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .prediction-graph {
            margin-top: 30px;
            border: 1px solid #0B2F9F;
            border-radius: 8px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
            animation: fadeInUp 1.5s ease;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .metric-container {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            animation: fadeIn 2s ease;
        }
        .metric-box {
            background-color: #0B2F9F;
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
            text-align: center;
            font-size: 18px;
            transition: transform 0.3s ease;
        }
        .metric-box:hover {
            transform: scale(1.1);
            background-color: #161D6F;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()
# Predefined list of stocks
predefined_stocks = {
    "Stock 1": "result_dataset.csv",
    "Stock 2": "result_dataset_more_cols_of_one_company.csv",
    "Stock 3": "microsoft_stock_data.csv",
    "Stock 4": "alphabet_stock_data.csv",
    "Stock 5": "amazon_stock_data.csv",
    "Stock 6": "apple_stock_data.csv",
    "Stock 7": "jpm_stock_data.csv",
    "Stock 8": "netflix_stock_data.csv",
    "Stock 9": "nvidia_stock_data.csv",
    "Stock 10": "meta_stock_data.csv",
    "Stock 11": "tesla_stock_data.csv",
    

    # Add paths to other stock CSVs
}

# Main title
st.title("Stock Price Prediction using LSTM Model")

# Dropdown for selecting a predefined stock
selected_stock = st.selectbox("Select a stock", list(predefined_stocks.keys()))

# Load and initialize the LSTM model
target_col = "Adj Close"
model_file = "lstm_trained_model.pkl"

if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    st.write("Loaded pre-trained model.")
else:
    st.write("Training the model initially...")
    sample_data = np.random.rand(5, 100)
    sample_target = np.random.rand(1, 100)
    model = LSTMCell(input_size=sample_data.shape[0], hidden_size=100)
    model.train(sample_data, sample_target, epochs=1000)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    st.write("Model trained and saved for future use.")

# Load the selected stock data
if selected_stock:
    file_path = predefined_stocks[selected_stock]
    df = pd.read_csv(file_path)
    df = df.dropna()
    st.write(f"### Data Preview for {selected_stock}:", df.head())

    # Select input columns for prediction
    input_cols = st.multiselect("Select Input Columns", df.columns, default=["Open", "High", "Low", "Close", "Volume"])

    if input_cols:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Scaling data
        input_data = df[input_cols].values
        target_data = df[[target_col]].values
        input_scaled = scaler_x.fit_transform(input_data).T
        target_scaled = scaler_y.fit_transform(target_data).T

        # Model prediction
        predictions_scaled = model.predict(input_scaled).T
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
        actuals = target_data[:len(predictions)].flatten()

        # Filter out NaN values
        mask = ~np.isnan(predictions) & ~np.isnan(actuals)
        actuals, predictions = actuals[mask], predictions[mask]

        # Plotting
        st.write("### Prediction Results")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actuals, label="Actual", color="#0B2F9F")
        ax.plot(predictions, label="Predicted", color="orange")
        ax.set_title("Actual vs Predicted Adjusted Closing Prices")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Adjusted Closing Price")
        ax.legend()
        st.pyplot(fig)

        # Model accuracy metrics
        st.write("### Model Accuracy Metrics")
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**R-squared (R²):** {r2:.4f}")
else:
    st.info("Select a stock to view its predictions.")
