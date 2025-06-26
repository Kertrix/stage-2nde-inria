import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(data, scale=True):
    data = pd.get_dummies(data)
    
    if scale:
        scaler = StandardScaler()
    
        data = scaler.fit_transform(data)
        print(data.shape)
    return data

def get_train_test_data(file_path, y_column, encode=True, scale=True):
    X_data = pd.read_csv(file_path)
    S = X_data["race"]
    y_data = X_data.pop(y_column)   
    
    if encode:
        X_data = prepare_data(X_data, scale=scale)
    
    return train_test_split(X_data, y_data, S, test_size=0.3, random_state=42)

    