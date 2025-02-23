# 📈 Stock Price Prediction using LSTM  

This repository contains an **LSTM-based Stock Price Prediction** system with a **Streamlit frontend** for visualization.  

## 🚀 Project Overview  
This project leverages **Long Short-Term Memory (LSTM) networks** to predict stock prices based on historical data. The **frontend**, built with **Streamlit**, provides an interactive UI for users to select stocks, view historical data, and visualize predictions.  

## 📂 Files in this Repository  
### 🔹 Code Files  
- `SRC.ipynb` – Jupyter Notebook containing the training pipeline for the LSTM model.  
- `stock_prediction_frontend.py` – A Streamlit-based application for stock price prediction.  

### 🔹 Stock Data Files  
- `alphabet_stock_data.csv`  
- `amazon_stock_data.csv`  
- `microsoft_stock_data.csv`  
- `netflix_stock_data.csv`  
- `nvidia_stock_data.csv`  
- `tesla_stock_data.csv`  
- `result_dataset.csv`  
- `result_dataset_more_cols_of_one_company.csv`  
- `total_closing_price.csv`  

## 📌 Features  
✅ **Pre-trained LSTM Model** (or trains a new one if unavailable)  
✅ **Interactive Stock Selection** (from predefined stock datasets)  
✅ **Dynamic Data Visualization** (Stock charts & prediction graphs)  
✅ **Evaluation Metrics** (MSE, MAE, R²)  

## 🛠️ Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/mr-coder07/project_1.git
   cd project_1

 ## 🛠️ Installation dependencies
	pip install -r requirements.txt

 ## Run the Streamlit app
 	streamlit run stock_prediction_frontend.py
## 📊 Usage
- `Select a stock from the dropdown.`
- `View historical stock data.`
- `Generate LSTM-based price predictions.`
- `Analyze prediction performance with evaluation metrics.`

