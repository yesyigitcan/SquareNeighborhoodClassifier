import pandas
#from Model import SquareNeighborhoodClassifier
from SquareNeighborhoodClassifier import SquareNeighborhoodClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

X_train = pandas.DataFrame(datasets.load_breast_cancer().data)[[0,1]]

y_train = datasets.load_breast_cancer().target

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42) 

model = SquareNeighborhoodClassifier(14, epoch=0, onlySetNone=True)
model.fit(X_train, y_train)
print("SNC Accuracy Score:", model.score(X_test, y_test))

model2 = KNeighborsClassifier(n_neighbors=225)
model2.fit(X_train, y_train)
print("KNN Accuracy Score:", model2.score(X_test, y_test))
