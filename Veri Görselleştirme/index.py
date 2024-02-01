import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x,y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

model = SVC()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk: ", accuracy)