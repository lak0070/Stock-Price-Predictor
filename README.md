# Stock-Price-Predictor

Overview of the script:

It fetches historical stock data from Yahoo Finance for a user-input stock ticker starting from April 1, 2010, up to today.The script selects the adjusted closing prices, scales them between 0 and 1, and segments them into sequences of 60 days to predict the next day's price.It constructs an LSTM deep learning model with two LSTM layers and dense layers for regression.The model is trained on 80% of the data for one epoch and tested on the remaining 20%.It predicts stock prices for the test set and calculates the root mean square error (RMSE) to evaluate performance. A plot shows the actual vs predicted closing prices. Finally, it predicts the next day's closing price based on the most recent 60 days of data. This model uses sequential time-series data and deep learning to forecast stock price trends, leveraging the ability of LSTM to capture long-term dependencies in time series 

Features:

Additional features of the provided stock price prediction code include:
User Input Flexibility: The model can predict prices for any stock ticker symbol provided by the user at runtime, making it reusable for a broad range of stocks.
Data Integrity Checks: The code checks for the existence of the 'Close' column and whether the filtered data is empty, raising errors to prevent downstream issues.
Data Splitting Strategy: It uses a clear 80/20 split for training and testing datasets, ensuring the model is evaluated on unseen data for realistic performance assessment.
LSTM Model Design: The use of two sequential LSTM layers allows the network to capture both short and long-range dependencies in stock price sequences.
Batch Size Control: Training with a batch size of 1 allows for fine-grained weight updates on each training example, which can be helpful for volatile time series data.
Visualization: The code provides a clear, labeled matplotlib plot to visually compare training data, validation data, and model predictions, aiding interpretability.
Next Day Prediction: By using the last 60 days of actual data for a one-step ahead prediction, the script can be applied for short-term trading decisions or insights.
Use of Industry Standard Libraries: It leverages widely used libraries like pandas, numpy, sklearn, TensorFlow/Keras, and yfinance, which assures compatibility, scalability, and community support.
Scalability: The modular approach to data preprocessing, model building, training, and prediction makes it extensible for enhancements, such as changing model architecture or including additional features like volume or technical indicators.
These additional aspects enhance the robustness, adaptability, and practical usability of the stock price prediction code for users interested in financial forecasting with machine learning

Technology and Tools used:

Python programming language is the base.
Pandas library for data manipulation and handling time series stock data.
Matplotlib for plotting price history and prediction visuals.
yfinance library to download historical stock market data.
Math and datetime modules for calculations and date handling.
NumPy library for numerical operations and array manipulation.
scikit-learn's MinMaxScaler for normalizing stock price data before training.
TensorFlow with Keras API used to build the Sequential LSTM deep learning model.
LSTM layers capture time series dependencies.
Dense layers for output regression.
Adam optimizer for model training with mean squared error loss function.
The model is trained on 80% of data and tested on the remaining 20%, with prediction performance evaluated by RMSE.
This stack combines data handling, visualization, and advanced deep learning techniques, typical for stock price forecasting projects using LSTM networks in Python.​
If you want, I can further explain how each part works or suggest improvements.

Installation Steps:

Install Python 3 if not already installed. Python 3.8 or later is recommended.
Install required Python libraries with pip:

Running the Project:

Save your Python script with the provided code to a file, e.g., stock_price_prediction.py.
Run the script in your terminal or command prompt.
When prompted, enter the stock ticker symbol
The script will download historical stock data, train the LSTM model, plot the results, and finally print the predicted next day close price.

Screenshorts:


