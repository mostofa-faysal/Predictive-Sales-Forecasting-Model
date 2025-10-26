"""
Predictive Retail Sales & Inventory Forecasting System
Author: Mostofa Faysal
Purpose: demonstrate end-to-end data-driven forecasting
"""


from src.data_preprocessing import load_data, preprocess_data
from src.forecasting_model import train_model, evaluate_model
from src.utils import plot_feature_importance
import pandas as pd

# Load data
data = load_data("C:/Users/User/Downloads/CO_OP/Projects/Predictive Sales Forecasting Model (Python + Power BI)/data/raw/retail_store_inventory.csv")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train
model = train_model(X_train, y_train)

# Evaluate
mae, r2 = evaluate_model(model, X_test, y_test)
print(f"MAE: {mae:.2f}, R^2: {r2:.2f}")

# Plot feature 
plot_feature_importance(model, X_train.columns)