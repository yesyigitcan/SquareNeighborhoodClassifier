import numpy
import pandas
import matplotlib.pyplot as plt
from copy import deepcopy

# SNC
class SquareNeighborhoodClassifier:
    def __init__(self, size = 4, offset = 0.0001):
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.predictList = []
        self.size = size
        self.offset = offset
        


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.ncol = len(X.columns)
        self.cols = list(X.columns)
        self.create()

    # Create min dictionary
    def min(self):
        self.minDict = {}
        for col in self.cols:
            self.minDict.update({col:numpy.min(self.X[col])})

# Fit part starts here
    # Create max dictionary    
    def max(self):
        self.maxDict = {}
        for col in self.cols:
            self.maxDict.update({col:numpy.max(self.X[col])})

    def range(self):
        self.rangeDict = {}
        for col in self.cols:
            rng = (self.maxDict[col] + self.offset - self.minDict[col])/float(self.size)
            self.rangeDict.update({col:rng})


    # Create squares
    def create(self):
        self.min()
        self.max()
        self.range()

        row = [[] for i in range(self.size)]
        self.squares = [deepcopy(row) for i in range(self.size)]
        self.squares_cvalue = deepcopy(self.squares)
        
        col_x, col_y = self.cols[0], self.cols[1]
        min_x, min_y = self.minDict[col_x], self.minDict[col_y]
        range_x, range_y = self.rangeDict[col_x], self.rangeDict[col_y]
        for index, row in self.X.iterrows():
            row_x, row_y = row[col_x], row[col_y]
            target = self.y[index]
            i, j = int((row_x - min_x) / range_x), int((row_y-min_y) / range_y)
            self.squares[i][j].append([row_x, row_y, target])
        
        for i in range(self.size):
            for j in range(self.size):
                c_x, c_y = min_x + range_x  * (i + 0.5), min_y + range_y  * (j + 0.5) 
                self.squares_cvalue[i][j].append(c_x)
                self.squares_cvalue[i][j].append(c_y)
                valueDict = {}
                countDict = {}
                totalCount = 0
                setFlag = 0
                for row_wtar in self.squares[i][j]:
                    target=row_wtar[2]
                    totalCount += 1
                    if target not in countDict:
                        setFlag = 1
                        countDict.update({target:0.0})       
                for row_wtar in self.squares[i][j]: # row with target
                    target=row_wtar[2]
                    if target not in valueDict:
                        valueDict.update({target:0.0})

                    valueDict[target] += self.euclidean_distance(row_wtar[:2], self.squares_cvalue[i][j][:2]) / countDict[target]
                    
                if setFlag:
                    self.squares_cvalue[i][j].append(min(valueDict, key=valueDict.get))
                else:
                    self.squares_cvalue[i][j].append(None)
                self.squares_cvalue[i][j].append(totalCount)

    def euclidean_distance(self, input1 , input2):
        temp = 0.0
        for i in range(len(input1)):
            temp += numpy.abs(input1[i]-input2[i])**2
        return numpy.sqrt(temp)
        

# Fit part ends here
# Predict parts starts here
    def predict(self, X_test):
        self.predictList = []
        self.X_test = X_test
        col_x, col_y = self.cols[0], self.cols[1]
        
        for index, row in self.X_test.iterrows():
            sq_pos_x, sq_pos_y = self.findSqPos([row[col_x], row[col_y]])
            g = self.guess(sq_pos_x, sq_pos_y, 0)
            if g != None:
                self.predictList.append(g)
            else:
                self.predictList.append('NO GUESS')
        return self.predictList

    def findSqPos(self, row):
        col_x, col_y = self.cols[0], self.cols[1]
        row_x, row_y = row[0], row[1]
        i, j = int((row_x - self.minDict[col_x]) / self.rangeDict[col_x]), int((row_y-self.minDict[col_y]) / self.rangeDict[col_y])
        if i < 0:
            i = 0
        elif  i >= self.size:
            i = self.size - 1
        if j < 0:
            j = 0
        elif j >= self.size:
            j = self.size - 1
        return i, j

    def square(self, sq_pos_x, sq_pos_y):
        return self.squares[sq_pos_x][sq_pos_y]  

    def guess(self, pos_x, pos_y, n=0):
        if n >= self.size:
            return None
        guessList = []
        for cur_x in range(pos_x-n, pos_x+n+1):
            for cur_y in range(pos_y-n, pos_y+n+1):
                if 0 <= cur_x < self.size and 0 <= cur_y < self.size:
                    g = self.squares_cvalue[cur_x][cur_y][2]
                    count = self.squares_cvalue[cur_x][cur_y][3]
                    if g != None:
                        guessList.append([g, count])
        if len(guessList) == 0:
            return self.guess(pos_x, pos_y, n+1)
        uniqueDict = {}
        for g in guessList:
            if g[0] not in uniqueDict:
                uniqueDict.update({g[0]:0})
            uniqueDict[g[0]] += g[1]
        return max(uniqueDict, key=uniqueDict.get)

    def score(self, y_test):
        true = 0
        total = 0
        for i in range(len(self.predictList)):
            if self.predictList[i] != 'NO GUESS':
                if self.predictList[i] == y_test[i]:
                    true += 1
            total += 1
        return true/float(total)
# Predict part ends here
# Plot part starts here
    def plot(self, mode='Train'):
        if len(self.predictList) == 0:
            raise Exception('Make Some Prediction Before Plot')
        if mode not in ('Train', 'Test', 'All'):
            raise Exception('Mode Not Found: ' + mode)
        col_x, col_y = self.cols[0], self.cols[1]
        min_x, min_y = self.minDict[col_x], self.minDict[col_y]
        max_x, max_y = self.maxDict[col_x], self.maxDict[col_y]
        range_x, range_y = self.rangeDict[col_x], self.rangeDict[col_y]
        curr_x = min_x
        curr_y = min_y
        for i in range(self.size+1):
            plt.plot([min_x, max_x], [curr_y, curr_y], 'g')
            curr_y += range_y
        for i in range(self.size+1):
            plt.plot([curr_x, curr_x],[min_y, max_y], 'g')
            curr_x += range_x
        for i in range(self.size):
            for j in range(self.size):
                square_target = self.squares_cvalue[i][j][2]
                if square_target == None:
                    square_target = "NC"

                plt.plot(self.squares_cvalue[i][j][0], self.squares_cvalue[i][j][1], 'go')
                plt.text(self.squares_cvalue[i][j][0], self.squares_cvalue[i][j][1], 'SC:' + str(square_target))
        if mode == 'Train' or mode == 'All':
            for index, row in self.X.iterrows():
                target = self.y[index]
                clr = 'bo'
                if target == 0:
                    clr = 'ro'
                plt.plot(row[col_x], row[col_y], clr)             
        elif mode == 'Test':
            for index, row in self.X_test.iterrows():
                target = self.y[index]
                clr = 'bo'
                if target == 0:
                    clr = 'ro'
                plt.plot(row[col_x], row[col_y], clr)
                plt.text(row[col_x], row[col_y], "G:" + str(self.predictList[index]))
        if mode == 'All':
            for index, row in self.X_test.iterrows():
                target = self.y[index]
                clr = 'co'
                if target == 0:
                    clr = 'mo'
                plt.plot(row[col_x], row[col_y], clr)
                plt.text(row[col_x], row[col_y], "G:" + str(self.predictList[index]))
        plt.title("SC | Square Center   NC | No Class Prediction    G | Guess")
        plt.show()

if __name__=='__main__':
    path = 'covid.csv'
    df = pandas.read_csv(path)
    
    X_train = df.drop('target', axis=1)
    y_train = df['target']

    test_set = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
              [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
              [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
              [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
              [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]
    df_test = pandas.DataFrame(test_set, columns = ['cough_level', 'fever', 'target'])
    X_test =  df_test.drop('target', axis=1)
    y_test = df_test['target']



    model = SquareNeighborhoodClassifier(size=4)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print(predict)
    print("Accuracy Score:", model.score(y_test))
    model.plot(mode='All')
