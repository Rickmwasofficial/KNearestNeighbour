from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

#split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

print(X_train.shape)
print(X_test.shape)

#Build the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

#train
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Check the accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

#test using dummy data
sample = [[1, 3, 5, 4], [1, 1, 1, 1]]
prediction = knn.predict(sample)
pred = [iris.target_names[p] for p in prediction]

print(f"Predictions: {pred}")
