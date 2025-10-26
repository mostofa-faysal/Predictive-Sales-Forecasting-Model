from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model(X_train, y_train):
    """Train XGBoost model"""
    model = XGBRegressor(n_estimator=500, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.joblib')
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model with MAE and R^2"""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mae, r2