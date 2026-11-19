import pandas as pd
import numpy as np

from perceptron.perceptron import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_perceptron_NAND_operation():
    # Input data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    # Labels
    y = np.array([1, 1, 1, 0])

    # Initialize Perceptron
    model = Perceptron()
    model.fit(X, y)


    # Predict the class labels for new data
    new_data = np.array([[1, 0], [1, 1], [0, 0]])
    y_pred = model.predict(new_data)
    assert np.array_equal(y_pred,np.array([1,0,1]))

def test_perceptron_as_linear_classifier():
    df = pd.read_csv("C:\\Scripts\\projects\\ai-engineering-projects\\perceptron\\data\\data.csv")

    if df is None:
        raise FileNotFoundError("the file data.csv not found")

    X = df.drop(columns=["label"])
    y = df.get("label")

    model = Perceptron()

    model.fit(X=X.to_numpy(),y=y)

    test = np.array([
        [4.4441,6.8725],
        [8.5676,4.6457],
        [3.6747,6.7966],
        [8.1676,4.6457]
    ])

    y_pred = model.predict(X=test)

    assert np.array_equal(y_pred,np.array([1,0,1,0]))


def test_perceptron_on_real_data():
    # 1. Load and standardize features
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Perceptron(epochs=50)
    model.fit(X=np.array(X_train),y=np.array(y_train))

    y_pred = model.predict(X=np.array(X_test))

    assert np.mean(y_pred == y_test) >= 0.95
