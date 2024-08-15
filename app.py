from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Load the scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load the column names used during training
all_columns = joblib.load('feature_names.pkl')

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get the form data
        warehouse = request.form.get('warehouse')
        product_category = request.form.get('product_category')
        date = request.form.get('date')

        if not warehouse or not product_category or not date:
            return render_template('index.html', prediction="Please provide all required inputs.")
        
        # Process the date
        date = pd.to_datetime(date)
        year = date.year
        month = date.month
        day = date.day
        weekday = date.weekday()

        # Prepare the input data
        input_data = pd.DataFrame({
            'Warehouse': [warehouse],
            'Product_Category': [product_category],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'Weekday': [weekday]
        })

        # One-hot encode and align columns with training data
        input_data_encoded = pd.get_dummies(input_data, columns=['Warehouse', 'Product_Category'])
        
        # Ensure all required columns are present
        for col in all_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
        input_data_encoded = input_data_encoded[all_columns]
        
        # Scale the input data
        input_data_scaled = scaler_X.transform(input_data_encoded)
        
        # Predict
        prediction_scaled = model.predict(input_data_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
    
    return render_template('index.html', prediction=prediction)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
