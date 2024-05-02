from flaml import AutoML
import pandas as pd
from train_test_split import split_train_test

def train_model(X_train, y_train, settings):
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, **settings)
    return automl

def load_data():
    train_data = pd.read_csv('train_data.csv')
    X_train = train_data.drop('target_column', axis=1)
    y_train = train_data['target_column']
    return X_train, y_train

if __name__ == "__main__":
    X_train, y_train = load_data()
    automl_settings = {
        "time_budget": 50,  # seconds
        "metric": 'r2',
        "task": 'regression',
        "estimator_list": ['rf']
    }
    model = train_model(X_train, y_train, automl_settings)
    print("Training complete.")
    print("Best model and its performance:")
    print(model.model.estimator)
