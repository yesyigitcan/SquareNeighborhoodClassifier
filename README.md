# Square Neighborhood Classifier

Let's say we have a universe that has limited size by minimum and maximum value of two table attributes.<br>
We assume this universe is a big square and divide it into smaller squares like we do in convolutional operation.<br>
Each small square has own class which is determined by a kind of weighted (by count) euclidean distance.<br>
<br>
Now let's assume we have a data row in order to predict its class.<br>
We find its square position in universe first.<br>
Then check if its square has a determined class.<br>
If it does, so the class of new data is the class of its square.<br>
Else let's increase our visibility distance and search for neighbor squares and make a prediction by using neighbor squares' opinions like we do in random forest.<br>
We have weightly determined classes in each square. But it is not enough. We also check count in each square to make another weighted prediction.<br> 
