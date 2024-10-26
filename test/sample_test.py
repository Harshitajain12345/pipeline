import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

# Sample test file for the app.py functionality

@pytest.fixture
def load_data():
    df = pd.read_csv('tips.csv')
    return df

def test_no_null_values(load_data):
    df = load_data
    assert df.isnull().sum().sum() == 0, "Data contains null values"

def test_label_encoding(load_data):
    df = load_data
    lb = LabelEncoder()
    df['sex'] = lb.fit_transform(df['sex'])
    df['smoker'] = lb.fit_transform(df['smoker'])
    df['day'] = lb.fit_transform(df['day'])
    df['time'] = lb.fit_transform(df['time'])

    # Checking if the columns are properly encoded (all values should be 0 or 1 for binary categories)
    assert set(df['sex'].unique()) == {0, 1}, "Sex column not properly encoded"
    assert set(df['smoker'].unique()) == {0, 1}, "Smoker column not properly encoded"
    assert df['day'].nunique() <= 7, "Day column encoding issue"
    assert df['time'].nunique() <= 2, "Time column encoding issue"

def test_model_accuracy(load_data):
    df = load_data
    lb = LabelEncoder()
    df['sex'] = lb.fit_transform(df['sex'])
    df['smoker'] = lb.fit_transform(df['smoker'])
    df['day'] = lb.fit_transform(df['day'])
    df['time'] = lb.fit_transform(df['time'])

    x = df.drop(columns=['total_bill'])
    y = df['total_bill']

    sc = StandardScaler()
    x_sc = sc.fit_transform(x)
    x_new = pd.DataFrame(x_sc, columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    assert r2 > 0.6, f"R2 score is too low: {r2}"

def test_shape_after_scaling(load_data):
    df = load_data
    lb = LabelEncoder()
    df['sex'] = lb.fit_transform(df['sex'])
    df['smoker'] = lb.fit_transform(df['smoker'])
    df['day'] = lb.fit_transform(df['day'])
    df['time'] = lb.fit_transform(df['time'])

    x = df.drop(columns=['total_bill'])
    sc = StandardScaler()
    x_sc = sc.fit_transform(x)

    assert x.shape == x_sc.shape, "Shape mismatch after scaling"

