This Dissertation focuses on pedestrian detection at night time. Three different Convolutional Neural Networks were implemented and tested to decide which one works best. The dataset provided was manually collected from google images. Three different sub-datasets were collected and marked accordingly in order to be used for training the networks.

### Sub-datasets used contained images of:
*  Pedestrians at night
* Landscapes
* Sheep

### Tools and languages used:
Python, Tensorflow

### Techniques implemented for better 

### Tests:
Run on: *NVidia GPU GTX 750Ti*
Multiples tests were run on these three CNNs and two varriables always changed.
-	Learning Rate: 
*[0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.0005, 0.001]*
-	Image Batch Size: 
*[1, 50, 100, 150, 200]*


