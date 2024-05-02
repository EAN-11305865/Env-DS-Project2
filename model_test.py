import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def load_test_data():
    test_data = pd.read_csv('test_data.csv')
    X_test = test_data.drop('target_column', axis=1)
    y_test = test_data['target_column']
    return X_test, y_test

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
