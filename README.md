This is the README for UNM CS529 Machine Learning Project 2.
It includes a command line interface for execution.

This was written by:
Anthony Galczak, Tristin Glunt
agalczak@unm.edu, tglunt@unm.edu

Required libraries for execution:
Python 3.6.0 & Anaconda 4.3.0 (pandas), SciPy 1.0.0 or higher (needed for load_npz)

Briefy summary on the files contained in this program:
naive_bayes.py
This includes all the functions needed for the Naive Bayes' algorithm. Has methods like determine_prior_probabilities that calculates P(Y) and determine_likelihoods that calculates P(X|Y).

logistic_regression.py
This includes all the functions needed for the Logistic Regression algorithm. Has methods like lr_train that does gradient descent and weight updates.

utilities.py
Miscellanous methods that are used between both algorithms and in main are found here. Includes methods like build_confusion_matrix, determine_most_important_features, and load_classes.

main.py
This includes the code for the command line interface and running our algorit
hms. An important part of main is this is where all the important parameters to tune are found such as beta, learning_rate, whether or not to show a confusion matrix. This is also where we load in all the files for training and testing.


TODO: Write about CLI, how to run, and cli examples.
