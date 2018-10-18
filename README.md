This is the README for UNM CS529 Machine Learning Project 2.
It includes a command line interface for execution.

This was written by:
Anthony Galczak, Tristin Glunt
agalczak@unm.edu, tglunt@unm.edu

Required libraries for execution:
Python 3.6.0 & Anaconda 4.3.0 (pandas), SciPy 1.0.0 or higher (needed for load_npz)

Brief summary on the files contained in this program:
naive_bayes.py
This includes all the functions needed for the Naive Bayes' algorithm. Has methods like determine_prior_probabilities that calculates P(Y) and determine_likelihoods that calculates P(X|Y).

logistic_regression.py
This includes all the functions needed for the Logistic Regression algorithm. Has methods like lr_train that does gradient descent and weight updates.

utilities.py
Miscellanous methods that are used between both algorithms and in main are found here. Includes methods like build_confusion_matrix, determine_most_important_features, and load_classes.

main.py
This includes the code for the command line interface and running our algorithms. An important part of main is this is where all the important parameters to tune are found such as beta, learning_rate, whether or not to show a confusion matrix, etc.. This is also where we load in all the files for training and testing.


We have two command line options when running main.py.
Type "python main.py -h" or "python main.py --help" to get help.

This will display the help menu shown below.

usage: main.py [-h] [-t] [-a [ALGORITHM]]

Anthony Galczak/Tristin Glunt
CS529 - Machine Learning
Project 2 - Naive Bayes' & Logistic Regression
NB & LR applied to "Bag of Words" document classification.

optional arguments:
  -h, --help      show this help message and exit
  -t              Set this to true if you would like to do tuning for the algorithm you specify.
  -a [ALGORITHM]  Which algorithm do you want to use. Acceptable options are "nb" or "lr".


The following are examples of how you can run the program:

python main.py -a nb -t
This will run the algorithm Naive Bayes' with tuning enabled (tuning Beta variable).

python main.py -a lr -t
This will run the algorithm Logistic Regression with tuning enabled (tuning Eta and Lambda variable).

python main.py -a nb
This will run the algorithm Naive Bayes' with no tuning. In other words, we will be training on the
full training dataset and run default parameters against the testing data.

python main.py
This will default to running Naive Bayes and no tuning.

This command line interface is not as full-featured as the last projects', but all of the relevant
parameters are available to be modified in main.py.


