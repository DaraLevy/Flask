import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

wine = load_wine()
X, y =  wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.02, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, y_test)
print(f"Accuarcy: {accuracy}")

with open('wine_model.pkl', 'wb') as f:
    pickle.dump(clf, f)