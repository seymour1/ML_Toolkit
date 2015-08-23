from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_digits
import numpy as np

mnist = load_digits()
clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, mnist.data, mnist.target)
print(scores.mean())

clf = ExtraTreesClassifier(n_estimators=10)
scores = cross_val_score(clf, mnist.data, mnist.target)
print(scores.mean())
