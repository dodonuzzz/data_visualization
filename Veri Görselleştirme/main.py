from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import make_classification

def load_data():
    x, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=5, random_state=42)
    return x, y

model = SVC()
param_dist = {"C": np.linspace(0.1, 100, 10), "gamma": np.linspace(0.001, 10, 10)}

rnd_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=25, cv=5, scoring="accuracy", random_state=42)

x,y = load_data()

rnd_search.fit(x,y)

best_params = rnd_search.best_params_
print("En iyi parametreler: ", best_params)
best_model = rnd_search.best_estimator_
print(best_model)