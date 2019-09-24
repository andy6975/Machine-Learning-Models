# Bayes Classifier

Bayesian Classifer is a simple and yet a very powerful algorithm. From classifying emails into spam and not spam to face recognition and emotion detection, this algorithm has stood the sands of time by still being a state-of-the-art algorithm. This repository is my take on it to understand its working and visualize its results.

## Just a little bit of theory before we dive into the mud!

![](./images/Bayes.jpg)

* The left hand side term is called ```aposteriori```, it denotes **the probability of class being w (omega) given an input vector x (bar)**. The class corresponding to its maximum is the assigned class for the data point x (bar).
* The first term in numerator of right hand side is called ```likelihood```, means **it is a function of parameters within the parameter space that describes the probability of obtaining the observed data** and is data dependent. This is excatly what we assume to be following Gaussian Distribution.
* The second term in numerator of right hand side is called ```priori```, equates **the probability of class w out of all the classes** and is also data dependent. If it is close to 0 or 1 then all the other terms are useless.
* The denominator on the right hand side is called ```evidence```, it can be viewed as merely a scale factor that guarantees that the ```aposteriori``` probabilities sum to one, as all good probabilities must.

We calculate ```aposteriori``` for each class for a given data point. The one having maximum is considered as the class of that data point.


## Application of Bayesian Classifier on different datasets:

### Data 1:

![](./Dataset_Train.png)