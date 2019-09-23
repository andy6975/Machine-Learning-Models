from evaluation import Evaluation
from data_preprocess import DataPreprocessing
from bayesian_classifier import BayesClassifier_Case1, BayesClassifier_Case2, BayesClassifier_Case3, BayesClassifier_Case4

# Enter the paths of each class data in corresponding variable
path_data_1 = '.../class1.txt'  # Path to class_1 data for a particular dataset
path_data_2 = '.../class2.txt'  # Path to class_2 data for a particular dataset
path_data_3 = '.../class3.txt'  # Path to class_3 data for a particular dataset

""" Instantiate an object of Data Preprocessing Class (read the documentation to understand what it does.) """

data_preprocess = DataPreprocessing(path_data_1, path_data_2, path_data_3, train_size=0.75)

""" Instantiate an object of Bayesian_Classifier Case (read the documentation for more insights.) """

bayes_classifier = BayesClassifier_Case1(data_prep=data_preprocess, train_size=0.70)
# bayes_classifier = BayesClassifier_Case2(data_prep=data_preprocess, train_size=0.70)
# bayes_classifier = BayesClassifier_Case3(data_prep=data_preprocess, train_size=0.70)
# bayes_classifier = BayesClassifier_Case4(data_prep=data_preprocess, train_size=0.70)

""" Plot the decision boundaries between required classes: (Read the documentation.) """

    # Between all three classes.
bayes_classifier.plot_contour(x_min=-500, x_max=2600, y_min=-100, y_max=3000, steps=750, pair_wise=None, train_or_test=0, smoothness=0.1, contour=False)

    # Between class 1 and class 2.
bayes_classifier.plot_contour(x_min=-500, x_max=2600, y_min=-100, y_max=3000, steps=750, pair_wise=[1, 2], train_or_test=0, smoothness=0.1, contour=False)

    # Between class 2 and class 3.
bayes_classifier.plot_contour(x_min=-500, x_max=2600, y_min=-100, y_max=3000, steps=750, pair_wise=[2, 3], train_or_test=0, smoothness=0.1, contour=False)

    # Between class 3 and class 1.
bayes_classifier.plot_contour(x_min=-500, x_max=2600, y_min=-100, y_max=3000, steps=750, pair_wise=[3, 1], train_or_test=0, smoothness=0.1, contour=False)

""" Instantiate an object of Evaluation class to calculate various model metrics. """

eval = Evaluation(bayes_case=bayes_classifier, data_prep=data_preprocess, test_size=0.30)
class_id = 1 # The class_id for the required class

    # Returns confusion matrix for a given Bayesian Classifier Case
cm = eval.confusion_matrix()

    # Returns the accuracy of classification for a given Bayesian Classifier Case
acc = eval.accuracy()

    # Returns the precision for a given class for a given Bayesian Classifier Case
prec = eval.precision(class_id)

    # Returns the recall for a given class for a given Bayesian Classifier Case
rec = eval.recall(class_id)

    # Returns the F-score for a given class for a given Bayesian Classifier Case
f_score = eval.f_score(class_id)

    # Returns the mean precision of classification for a given Bayesian Classifier Case
mean_prec = eval.mean_precision()

    # Returns the mean recall of classification for a given Bayesian Classifier Case
mean_rec = eval.mean_recall()

    # Returns the mean F-score of classification for a given Bayesian Classifier Case
mean_f_score = eval.mean_f_score()

    # Plots the confusion matrix of classification for a given Bayesian Classifier Case
eval.plot_confusion_matrix()