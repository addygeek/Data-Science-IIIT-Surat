from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
clf = RandomForestClassifier().fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))
