import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load dataset from CSV"""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Clean, feature engineer, and split data"""
    df = df.dropna(subset=['Date', 'Sales'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('date')

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['dayOfWeek'] = df['Date'].dt.dayofweek

    features = ['Month', 'Year', 'DayOfWeek', 'Price', 'Promo']
    target = 'Sales'

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)