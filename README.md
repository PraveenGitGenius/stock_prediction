# Stock Price Prediction Web Application

## 1. Introduction

The objective of this project is to develop a web-based application that can predict future stock prices using historical market data. The application is built using Python and Streamlit, providing an interactive and user-friendly interface for analysis and visualization.

This project focuses on understanding time-series forecasting and applying machine learning concepts to real-world financial data.

---

## 2. Problem Statement

Stock price prediction is a challenging task due to market volatility and various external influencing factors. The goal of this project is to analyze past stock data and generate future predictions using a forecasting model.

---

## 3. Technologies Used

- Python
- Streamlit (for web interface)
- Prophet (for time-series forecasting)
- Pandas & NumPy (data processing)
- yfinance (stock data collection)
- Plotly (interactive visualization)

---

## 4. Working of the Application

1. User selects a stock ticker symbol.
2. Historical data is fetched using the yfinance library.
3. Data is processed and formatted for the forecasting model.
4. Prophet model is trained on historical data.
5. Future stock prices are predicted.
6. Results are displayed using interactive graphs.

---

## 5. Project Structure

- app.py → Main application file
- requirements.txt → Required dependencies
- README.md → Project documentation

---

## 6. Installation and Setup

Step 1: Clone the repository

git clone https://github.com/PraveenGitGenius/stock_prediction.git

Step 2: Navigate to project directory

cd stock_prediction

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Run the application

streamlit run app.py

---

## 7. Key Learning Outcomes

- Understanding of time-series forecasting
- Hands-on experience with Streamlit
- Integration of APIs for real-time data
- Data visualization using Plotly
- Model training and prediction workflow

---

## 8. Limitations

- Predictions are based only on historical trends.
- Does not consider external economic factors.
- Market behavior can be highly unpredictable.

---

## 9. Future Scope

- Integration of multiple ML models for comparison
- Deployment on cloud platform
- Adding performance metrics
- Real-time news sentiment analysis

---

## Author

Praveen  
Electronics and Communication Engineering  
