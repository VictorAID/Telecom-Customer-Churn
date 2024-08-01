import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    numerical_feature = {feature for feature in data.columns if data[feature].dtypes != 'O'}
    categorical_feature = {feature for feature in data.columns if data[feature].dtypes == 'O'}

    encoder = LabelEncoder()
    for feature in categorical_feature:
        data[feature] = encoder.fit_transform(data[feature])
    
    data.drop(columns=['customerID'], inplace=True)
    
    X = data.drop(columns='Churn')
    y = data['Churn']
    
    return X, y

