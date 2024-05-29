import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st
from datetime import datetime, timedelta

# Load the saved model
loaded_model = pickle.load(open('rf_model.pkl', 'rb'))

# Load the sales data
data = pd.read_csv("train.csv")

# Preprocess your data (assuming dates are in DD/MM/YYYY format)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y', errors='coerce')
data.dropna(subset=['date'], inplace=True) # Remove rows with missing dates

# Preprocess your data
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# Separate features (X) and target variable (y)
X = data[['year', 'month', 'day', 'day_of_week', 'store', 'item']] 
y = data['sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale your features (optional but recommended for Random Forest)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.title("Sales Prediction App")

# Get user inputs
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# Get unique items and product names (replace with your actual data)
unique_items = data['item'].unique()
product_names = {
    1: "Bread", 
    2: "Sugar",
    3: "Salt",
    4: "Rice",
    5: "Flour",
    6: "Mealie Meal",
    7: "Spaghetti",
    8: "Biscuits",
    9: "Macaroni",
    10: "Cooking Oil",
}

# Create a mapping for item IDs to product names
item_to_product = {item: product_names.get(item, "Unknown Product") for item in unique_items}

# Create a dropdown for selecting products
product = st.selectbox("Select Product", list(item_to_product.values()))

# Get the item ID corresponding to the selected product
selected_item = [item for item, name in item_to_product.items() if name == product][0]

# Generate a date range for prediction
dates = pd.date_range(start=start_date, end=end_date)
predictions = []
total_predicted_sales = 0
if st.button("Predict Sales"):
    for date in dates:
        # Create a DataFrame for the prediction
        prediction_data = pd.DataFrame({'date': [date]})
        prediction_data['year'] = prediction_data['date'].dt.year
        prediction_data['month'] = prediction_data['date'].dt.month
        prediction_data['day'] = prediction_data['date'].dt.day
        prediction_data['day_of_week'] = prediction_data['date'].dt.dayofweek
        prediction_data['store'] = 1 # Assuming you want to forecast for store 1
        prediction_data['item'] = selected_item

        # Scale the input data
        scaled_data = scaler.transform(prediction_data[['year', 'month', 'day', 'day_of_week', 'store', 'item']])

        # Make prediction
        predicted_sales = loaded_model.predict(scaled_data)[0]

        # Adjust predictions based on item and add unit
        if selected_item in [2]:
            predicted_sales /= 8
            predicted_sales_str = f"{predicted_sales:.2f} kg"
        elif selected_item in [3, 4, 5, 6]:
            predicted_sales /= 4
            predicted_sales_str = f"{predicted_sales:.2f} kg"
        elif selected_item in  [7, 8, 9]:
            predicted_sales /= 4
            predicted_sales_str = f"{predicted_sales:.2f} packets"
        elif selected_item == 10:
            predicted_sales /= 5
            predicted_sales_str = f"{predicted_sales:.2f} litres"
        else:
            predicted_sales /= 2
            predicted_sales_str = f"{predicted_sales:.2f} loaves"
            
        predictions.append(predicted_sales_str)
        total_predicted_sales += predicted_sales
        

    # Display the predicted sales
    st.write(f"Predicted Sales for {product}:")
    st.dataframe(pd.DataFrame({'Date': dates, 'Predicted Sales': predictions}))
    st.write(f"Total predicted Sales: {total_predicted_sales:.2f}")