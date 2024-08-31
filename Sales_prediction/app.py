# app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\anshu\OneDrive\Desktop\sales_data.csv")

# Data preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])  # Drop rows where date conversion failed
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Features and Target
X = data[['Day', 'Month', 'Year']]
y = data['sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open('sales_model.pkl', 'wb'))

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = pickle.load(open('sales_model.pkl', 'rb'))
    
    # Get user input
    date_str = request.form['date']
    
    try:
        date = pd.to_datetime(date_str, format='%Y-%m-%d')  # Specify format if known
        day = date.day
        month = date.month
        year = date.year
    except Exception as e:
        return render_template('prediction.html', prediction=f"Error: {str(e)}")
    
    # Prediction
    prediction = model.predict([[day, month, year]])
    
    # Render the result page
    return render_template('prediction.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
