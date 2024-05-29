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
The model is assumed to be trained on the train.csv data.
The item column in the CSV should contain numerical IDs representing different products.
The product_names dictionary in the code maps these IDs to actual product names.
You might need to modify this dictionary based on your data.
The code assumes that you want to predict sales for store 1. 
You can change this in the prediction_data DataFrame.
 
Example
If your train.csv has the following data:
Date	Store	Item	Sales
01/01/2023	1	1	100
02/01/2023	1	1	120
03/01/2023	1	2	80
...	...	...	...
And your rf_model.pkl contains a trained model, running the app will allow you to predict sales for "Bread" (Item 1) or other products.