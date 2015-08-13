from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

mnist = load_digits(2)

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 2
bst = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(mnist.data, mnist.ta$
preds = bst.predict(mnist.data)

print preds