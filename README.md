#Sales Prediction App

1. This Streamlit app allows you to predict sales for 
Small scale retail business owners to predict sales for 
Different products based on a trained Random Forest model.

2. Requirements
Python 3.6 or higher
Streamlit
scikit-learn
pandas
pickle
Install the required packages using pip:
pip install streamlit scikit-learn pandas pickle

3. Setup
Train and save the model: You need to have already trained a Random Forest Regression model
and saved it as rf_model.pkl. Make sure this file is in the same directory as app.py.
Prepare the sales data: Make sure your sales data is in a CSV file named train.csv. 
The CSV should contain columns for date, store, item, and sales (adjust column names if needed).
Run the app: Open a terminal in the directory where you saved the files and run the following command:
streamlit run app.py


4. Usage
The app will open in your browser.
Enter the start and end dates for the prediction period.
Select the product you want to predict sales for.
Click the "Predict Sales" button.
The app will display a table showing the predicted sales for
Each day in the selected period, along with the total predicted sales.

5.Notes
The model is  trained on the train.csv data.
The item column in the CSV should contain numerical IDs representing different products.
The product_names dictionary in the code maps these IDs to actual product names.
You might need to modify this dictionary based on your data.
The code assumes that you want to predict sales for store 1. 
You can change this in the prediction_data DataFrame.
 
1. Sales Prediction App for Small-Scale Retailers in Zimbabwe
This repository contains the code for a Streamlit app that predicts sales for
small-scale retailers in Zimbabwe using a Random Forest machine learning model. 
The app is based on the research project documented in the provided project documentation.

2. Project Background
The research project aims to improve demand forecasting for
small-scale retailers in Zimbabwe by applying machine learning algorithms.
This app is a practical implementation of the project,
demonstrating the use of the Random Forest algorithm for retail forecasting.

3. How to Run the App
Install Dependencies:
pip install streamlit scikit-learn pandas pickle

Prepare Data:
Make sure you have a CSV file named train.csv containing sales data with the following columns:
date (DD/MM/YYYY format)
store (numerical store ID)
item (numerical product ID)
sales (sales quantity)

You'll also need a trained Random Forest model saved as rf_model.pkl.
Run the App:
streamlit run app.py

App Functionality
The app allows users to:
Input a start and end date for prediction.
Select a product from a dropdown list.
Generate predicted sales for the selected product and date range.
View the total predicted sales.
Key Components
app.py: Contains the Streamlit app code.
rf_model.pkl: The saved Random Forest model.
train.csv: The CSV file containing sales data.
Additional Notes
The product_names dictionary maps product IDs to product names. You may need to modify this based on your specific data.
The app assumes you want to predict sales for store 1. You can change this in the prediction_data DataFrame.
Future Improvements
Integrate additional features (economic indicators, competitive pricing, etc.) into the model.
Explore hyperparameter tuning to optimize model performance.
Develop real-time forecasting capabilities.
Create a more robust and scalable platform for data management and analysis.
This app provides a basic framework for implementing retail forecasting using Random Forest. It can be further customized and enhanced to create a comprehensive solution for small-scale retailers in Zimbabwe.
