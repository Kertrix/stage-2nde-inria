import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(data):
    data = pd.get_dummies(data)
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data

def get_train_test_data(file_path, y_column):
    X_data = pd.read_csv(file_path)
    y_data = X_data.pop(y_column)   
    
    X_data = prepare_data(X_data)
    
    return train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    