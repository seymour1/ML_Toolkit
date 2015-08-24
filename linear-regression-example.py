from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np

boston = load_boston()
boston_X_train = boston.data[:-20]
boston_X_test = boston.data[-20:]

boston_y_train = boston.target[:-20]
boston_y_test = boston.target[-20:]

model = LinearRegression()
model.fit(boston_X_train, boston_y_train)

print(model.score(boston_X_test, boston_y_test))
