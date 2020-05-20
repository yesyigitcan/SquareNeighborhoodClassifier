import pandas
from Model import SquareNeighborhoodClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#path = 'C:\\Users\\YigitCan\\Desktop\\Dersler\\Data Mining\\My Lab\\covid.csv'
path2 = 'C:\\Users\\YigitCan\\Desktop\\Tez-Workspace\\Dataset\\creditcard.csv'

#df = pandas.read_csv(path)
df = pandas.read_csv(path2)

#X_train = df.drop('target', axis=1)
#y_train = df['target']
X_train = df[['V1','V2']]
y_train = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42) 
'''
test_set = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
            [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
            [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
            [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
            [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]
df_test = pandas.DataFrame(test_set, columns = ['cough_level', 'fever', 'target'])
X_test =  df_test.drop('target', axis=1)
y_test = df_test['target']
'''

model = SquareNeighborhoodClassifier(size=6, epoch=5)
model.fit(X_train, y_train)
predict = model.predict(X_test)
# print(predict)
print("Accuracy Score:", model.score(y_test))
model.plot(mode='All', showValue=True)
# 3, 6 -> 0.8

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
print("KNN Accuracy Score:", model2.score(X_test, y_test))

# 13, 14 -> 0.8


