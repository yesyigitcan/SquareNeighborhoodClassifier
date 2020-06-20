# Square Neighborhood Classifier

<h2>Inspiration</h2>
If you ever played GTA San Andreas, you know there are some gangs that capture neighborhoods. These captured neighborhoods are represented as squares in different colors. SquareNeighborhoodClassifier works similarly.

<h2>Method</h2>
In order to use this algorithm, you can only use two features for now. Let's assume we want to make predictions by using two features. We have a huge square map (or universe) that is limited by minimum and maximum values of the features. Now we divide it into smaller squares like we do in convolutional operation. Each small square has own class (belongs to a specific gang) which is determined by a kind of weighted (by count) euclidean distance.

<h2>Prediction</h2>
Now let's assume we have a data row to predict its class(or which gang it belongs to)
<ol>
  <li>We find its square position in universe first.</li>
  <li>Then check if its square has a determined class.</li>
  <li>If it does, so the class of new data is the class of its square.</li>
  <li>Else let's increase our visibility distance and search for neighbor squares and make a prediction by using neighbor squares' opinions like we do in random forest.</li>
  <li>We have weightly determined classes in each square. But it is not enough. We also check count in each square to make another weighted prediction.</li>
</ol>

<h2>GTA San Andreas Map</h2>

!["GTA San Andreas Map"](gta-sa-map.webp)

<h2>Square Neighborhood Map</h2>

!["Square Neighborhood Map"](square-neighborhood-classifier.png)
