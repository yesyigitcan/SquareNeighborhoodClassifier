import numpy
import pandas
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import time

__all__ = ['SquareNeighborhoodClassifier']

# SNC
class SquareNeighborhoodClassifier:
    def __init__(self, size, epoch = 0, offset = 0.0001, onlySetNone = True):
        self.X = None
        self.y = None
        self.predictList = []
        self.size = size
        self.epoch = epoch
        self.offset = offset
        self.onlySetNone = onlySetNone
        


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
            self.minDict.update({col:self.X[col].min()})

# Fit part starts here
    # Create max dictionary    
    def max(self):
        self.maxDict = {}
        for col in self.cols:
            self.maxDict.update({col:self.X[col].max()})

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

        
        # Add data points into squares
        bf = '{desc}: {n_fmt}/{total_fmt} | Remaining: {remaining}'
        counter = 0
        for index, row in tqdm(self.X.iterrows(), total=len(self.X),desc="Data points are being added to initial squares", bar_format=bf):
            row_x, row_y = row[col_x], row[col_y]
            target = self.y[counter]
            i, j = int((row_x - min_x) / range_x), int((row_y-min_y) / range_y)
            self.squares[i][j].append([row_x, row_y, target])
            counter += 1
        
        # Calculate square value
        bf = '{desc}: {n_fmt}/{total_fmt}'
        progress = tqdm(total=self.size**2, desc="Calculating value of each square", bar_format=bf)
        for i in range(self.size):
            for j in range(self.size):
                progress.update(1)
                c_x, c_y = min_x + range_x  * (i + 0.5), min_y + range_y  * (j + 0.5) 
                self.squares_cvalue[i][j].append(c_x)
                self.squares_cvalue[i][j].append(c_y)
                valueDict = {}
                setFlag = 0 
                for row_wtar in self.squares[i][j]: # row with target
                    target=row_wtar[2]
                    if target not in valueDict:
                        setFlag = 1
                        valueDict.update({target:0.0})
                    valueDict[target] += 1.0 / self.euclidean_distance(row_wtar[:2], self.squares_cvalue[i][j][:2])
                if setFlag:
                    ass_target = max(valueDict, key=valueDict.get)
                    self.squares_cvalue[i][j].append(ass_target)
                    self.squares_cvalue[i][j].append(valueDict[ass_target])
                else:
                    self.squares_cvalue[i][j].append(None)
                    self.squares_cvalue[i][j].append(0.0)
        progress.close()
        # Update Square Value
        isAllSatisfy = 0
        epochCounter = 0
        progress = tqdm(total=self.epoch, desc="Epoch", bar_format=bf)
        while isAllSatisfy == 0 or epochCounter != self.epoch:
            progress.update(1)
            updated_squares_cvalue = deepcopy(self.squares_cvalue)
            isAllSatisfy = 1
            if epochCounter != self.epoch:
                epochCounter += 1
            else:
                progress.total = progress.total+1
                progress.refresh()
                progress.desc = "Epoch (extended)"
            
            for i in range(self.size):
                for j in range(self.size):
                    if self.squares_cvalue[i][j][2] != "sdasd":
                        value = 0.0
                        valueDict = {}
                        for k in range(i-1, i+2):
                            for l in range(j-1, j+2):
                                if 0 <= k < self.size and 0 <= l < self.size:
                                    target = self.squares_cvalue[k][l][2]
                                    if target == None:
                                        isAllSatisfy = 0
                                        continue
                                    if  target not in valueDict:
                                        valueDict.update({target:0.0})
                                    if i != k and j != l:
                                        valueDict[target] += self.squares_cvalue[k][l][3] /  (2 * numpy.sqrt(2))
                                    else:
                                        valueDict[target] += self.squares_cvalue[k][l][3] / 2
                        if len(valueDict ) == 0:
                            continue
                        ass_target = max(valueDict, key=valueDict.get)
                        updated_squares_cvalue[i][j][2] = ass_target
                        updated_squares_cvalue[i][j][3] = valueDict[ass_target]
            self.squares_cvalue = updated_squares_cvalue
        progress.close()

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
        return self.squares_cvalue[pos_x][pos_y][2]

    def score(self, X_test, y_test):
        pred = self.predict(X_test)
        y_testList = list(y_test)
        true = 0
        total = 0
        for i in range(len(pred)):
            if pred[i] != 'NO GUESS':
                if pred[i] == y_testList[i]:
                    true += 1
            total += 1
        return true/float(total)
# Predict part ends here
# Plot part starts here
    def plot(self, X_test, y_test, mode='Train', showValue=False):
        predictList = self.predict(X_test)
        if len(predictList) == 0:
            raise Exception('Make Some Prediction Before Plot')
        if mode not in ('Train', 'Test', 'Empty'):
            raise Exception('Mode Not Found: ' + mode)
        col_x, col_y = self.cols[0], self.cols[1]
        min_x, min_y = self.minDict[col_x], self.minDict[col_y]
        max_x, max_y = self.maxDict[col_x], self.maxDict[col_y]
        range_x, range_y = self.rangeDict[col_x], self.rangeDict[col_y]
        curr_x = min_x
        curr_y = min_y
        
        #plt.axhspan(min_y, min_y + range_y, facecolor='0.2', alpha=0.5)
        #plt.axhspan(min_y + range_y, min_y + 2 *range_y, facecolor='0.2', alpha=0.5)
        for i in range(self.size+1):
            plt.plot([min_x, max_x], [curr_y, curr_y], 'g')
            curr_y += range_y
        for i in range(self.size+1):
            plt.plot([curr_x, curr_x],[min_y, max_y], 'g')
            curr_x += range_x
        for i in range(self.size):
            for j in range(self.size):
                square_target = self.squares_cvalue[i][j][2]
                square_value = self.squares_cvalue[i][j][3]
                if square_target == None:
                    square_target = "NC"

                plt.plot(self.squares_cvalue[i][j][0], self.squares_cvalue[i][j][1], 'go')
                if showValue:
                    plt.text(self.squares_cvalue[i][j][0], self.squares_cvalue[i][j][1], str(square_target) + ' ' + '{:.2f}'.format(square_value))
                
        if mode == 'Train':
            for index, row in self.X.iterrows():
                target = self.y[index]
                clr = 'bo'
                if target == 0:
                    clr = 'ro'
                plt.plot(row[col_x], row[col_y], clr)             
        if mode in ('Train', 'Test'):
            y_testList = list(y_test)
            counter = 0
            for index, row in X_test.iterrows():
                target = y_testList[counter]
                clr = 'bo'
                if target == 0:
                    clr = 'ro'
                plt.plot(row[col_x], row[col_y], clr)
                plt.text(row[col_x], row[col_y], "G:" + str(predictList[counter]))
                counter += 1
        plt.title("SC | Square Center   NC | No Class Prediction    G | Guess")
        plt.show()


