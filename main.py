

import pandas as pd
import scipy.sparse
import numpy as np
import math
import time
import copy
import operator
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

"""
    Project 2 CS 529 - Naive Bayes and Logistic Regression from scratch

    @authors:
        Tristin Glunt | tglunt@unm.edu
        Anthony Galczak | agalczak@unm.edu

    Required Libraries:
        - SciPy 1.0.0 (loading the npz format as a csr_matrix) or higher
"""

num_of_classes = 20

def main():

    # Loads in a sparse matrix (csr_matrix) from a npz file.
    training_data = scipy.sparse.load_npz("training_sparse.npz")

    # Loading the testing data from an npz file also.
    test_data = scipy.sparse.load_npz("testing_sparse.npz")



    # Tuning our naive bayes' given a range of Beta variables.
    # TODO: Write an if statement that can be used here at the command line
    # to do tuning or testing solution.
    # TODO: We could pass in the beta variables/range from main....

    # Splits our data into training data and validation data.
    #X_train, X_validation = train_test_split(training_data, test_size = .2, shuffle = True)
    #nb_tuning(X_train, X_validation, test_data)

    # TODO: Write Naive Bayes solution for testing data. Should be a tiny
    # method that trains, predicts and outputs.

    #logistic_regression_solution(X_train, X_validation, test_data)

    nb_solve(training_data, test_data)



# nb_tuning()
# Tunes naive bayes for a range of Beta values. This method will run the Naive Bayes'
# algorithm for each of these Beta variables and then plot accuracy vs. the validation
# data set when it is done running.
def nb_tuning(X_train, X_validation, test_data):
    print("Training set size: " + str(X_train.shape))
    print("Validation set size: " + str(X_validation.shape))

    X_validation_classification = X_validation[:, -1:]

    # Beta is the tuning term for Laplace smoothing
    betas = [.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1]
    accuracies = []

    # Go through and train on each beta
    # TODO: Store likelihood_probabilities and prior_probabilities that best
    # classify the validation data and then use this to train
    for beta in betas:
        # Training naive bayes
        likelihood_probabilities, prior_probabilities = nb_train(X_train, beta)
        predictions = nb_predict(X_validation, prior_probabilities, likelihood_probabilities)

        accuracy = 0
        for i in range(X_validation.shape[0]):
            if(predictions[i] == X_validation_classification[i]):
                accuracy += 1
        accuracy /= X_validation.shape[0]
        print("Accuracy on validation set with beta "  + str(beta) + " : "+ str(accuracy))
        accuracies.append(accuracy)

    # Generating plots for beta variable vs. accuracy on validation data.
    print(betas)
    print(accuracies)
    plt.semilogx(betas, accuracies, linewidth=2.0)
    plt.xlabel('Beta')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Validation Data while tuning Beta parameter')
    plt.show()

    output_predictions("validation_output.csv", predictions, X_train.shape[0])

# nb_solve()
# Training our naive bayes' algorithm against our full set of training data and
# then getting predictions on testing data and then outputting that to a file.
def nb_solve(training_data, testing_data):
    beta = .01 # TODO: Hard-coded beta for now.
    likelihood_probabilities, prior_probabilities = nb_train(training_data, beta)
    predictions = nb_predict(testing_data, prior_probabilities, likelihood_probabilities)
    output_predictions("testing_predictions.csv", predictions, training_data.shape[0])

# nb_train()
# Meta method for building P(Y) and P(X|Y) probabilities from Naive Bayes.
# This method will bring in a set of data (training data, typically separated from
# validation data), and a Beta tuning variable.
def nb_train(data, beta):
    # returns a tuple of lists that contain the non-zero indexes of the matrix data ([row_indices], [col_indices])
    non_zero_data = data.nonzero()

    # Loading in classes as strings from newsgrouplabels.txt
    classes = load_classes("newsgrouplabels.txt")

    # Calculate total # of words per a class. Needed for determine_likelihoods.
    total_words_in_class = determine_total_words_in_classes(data)

    # Calculate the ratio of prior probabilities, i.e. given_class/total_examples
    prior_probabilities= determine_prior_probabilities(data[:, -1:])

    # pass the dataset except the classifications
    likelihood_probabilities = determine_likelihoods(data, non_zero_data, total_words_in_class, beta)

    return likelihood_probabilities, prior_probabilities

# nb_predict()
# Calculates the prediction function for Naive Bayes
# Classifies a set of data (validation or testing) based upon the likelihood
# matrix (P(X|Y)) and priors (P(Y)) that we calculated earlier.
def nb_predict(data, prior_probabilities, likelihood_probabilities):

    log_priors = []
    for value in prior_probabilities.values():
        log_value = math.log(value)
        log_priors.append(log_value)

    likelihood_probabilities.data = np.log(likelihood_probabilities.data)
    # gives matrix of (examples, classes)
    print(likelihood_probabilities.shape)
    print(data.shape)

    new_data_dotted_likelihoods = data.dot(np.transpose(likelihood_probabilities))
    new_data_dotted_likelihoods_plus_priors = new_data_dotted_likelihoods + log_priors

    print("New data dotted with likelihood: " + str(new_data_dotted_likelihoods_plus_priors.shape))

    # take maximums of each example (go through every class for an example and find the max)
    maximum_indices_for_each_example = new_data_dotted_likelihoods_plus_priors.argmax(axis=1)
    print("Maximum indices" + str(maximum_indices_for_each_example.shape))

    predictions = []
    for index in maximum_indices_for_each_example:
        predictions.append(index + 1)
    print("Predictions shape: " + str(np.array(predictions).shape))

    return predictions

def determine_most_important_features(likelihood_probabilities):
    # take the sum of each column
    total_probabilities = likelihood_probabilities.sum(axis=0)
    # indices of top 1000 totals
    ind_total_prob = np.argpartition(total_probabilities, -60000)[-60000:]
    print(len(ind_total_prob))
    print(ind_total_prob)
    return ind_total_prob

# determine_total_words_in_classes()
# Calculating how many total words are in each classification.
# This is useful in likelihood calculation in denominator.
def determine_total_words_in_classes(data):

    total_examples = data.shape[0]
    # We don't want the class counts to interfere with data counts
    classifications = data[:,-1:]
    data_without_classes = data[:,0:-1]

    # Get the sum of each row, this returns a column vector
    row_sums = data_without_classes.sum(axis=1)

    # Initializing 20 dictionary elements for each newsgroup
    total_words_in_class = {}
    for x in range(num_of_classes):
        total_words_in_class["class" + str(x)] = 0

    for x in range(total_examples):
        current_class = (classifications.data[x] - 1) # NOTE: The classifications are 1-index'ed. This is "the fix"
        total_words_in_class["class" + str(current_class)] += row_sums[x][0]

    return total_words_in_class


# determine_prior_probabilities()
# This calculates the prior ratio's of a given class / total examples.
# i.e. "alt.atheism" has 490 words out of 18900 words total.
# Returns a dictionary of the prior probabilities (Represented in formula by P(Y))
def determine_prior_probabilities(classifications):

    class_counts = {}
    prior_probabilities = {}

    # initialize class counts for dictionary
    for i in range(num_of_classes):
        class_counts["class" + str(i)] = 0

    # add 1 for every label you encounter (1 instance)
    for label in classifications.data:
        class_counts["class" + str(label - 1)] += 1 # NOTE: The classifications are 1-index'ed. This is "the fix"

    # calculate the prior probabilities by dividing each class count by the total examples
    for i in range(num_of_classes):
        prior_probabilities["class" + str(i)] = class_counts["class" + str(i)] / len(classifications.data)

    return prior_probabilities


# determine_likelihoods()
# build a matrix: (classes, features) -> value is P(X|Y)
# return matrix of probabilites
# calculate P(X|Y) -> count # words in feature i with class k / total words in class k
def determine_likelihoods(data, non_zero_data, total_words_in_class, beta):

    likelihood_matrix = np.zeros((num_of_classes, 61189)) # NOTE: 61189 because we have classifications also.
    length_of_nonzero_data = len(non_zero_data[0])

    # saving current row saves us ~1.5m hits for the entire data
    current_row_index = -1
    for i in range(length_of_nonzero_data):

        # getting coordinates of nonzero ele
        row_index = non_zero_data[0][i]
        col_index = non_zero_data[1][i]

        #if we're dealing with a new row
        if(row_index != current_row_index):
            current_classification = (data[row_index, -1:].data[0]) - 1 # NOTE: The classifications are 1-index'ed. This is "the fix"
            current_row_index = row_index

        current_val = data[row_index, col_index]

        likelihood_matrix[current_classification][col_index] += current_val

    # Now that we have looped over all the non-zero data, we need to add laplace
    # (1/61188) and divide it all by "total all words in Yk + 1"
    for x in range(num_of_classes):
        total_words = total_words_in_class["class" + str(x)]
        for y in range(61189):
            enhanced_likelihood = likelihood_matrix[x][y]
            enhanced_likelihood += beta
            enhanced_likelihood /= (total_words + (61188 * beta))
            likelihood_matrix[x][y] = enhanced_likelihood

    return likelihood_matrix

# logistic_regression_solution: preprocessing and steps needed to use the logitic reg. alg
# Trains using Gradient descent
def logistic_regression_solution(X_train, X_validation, test_data):
    likelihood_probabilities, prior_probabilities = nb_train(X_train, 0.001)
    array_of_most_imp_features = determine_most_important_features(likelihood_probabilities)

    spliced_X_train = scipy.sparse.csr_matrix(X_train.tocsc()[:, array_of_most_imp_features])
    spliced_X_validation= scipy.sparse.csr_matrix(X_validation.tocsc()[:, array_of_most_imp_features])
    print(spliced_X_train.shape)

    # separate features and classifications
    X_train_data = spliced_X_train[:, :-1]
    X_train_classifications = X_train[:, -1:]

    X_validation_data = spliced_X_validation[:, :-1]
    X_validation_classification = X_validation[:, -1:]

    # TODO: figure out dimensionality reduction techinique
    # truncated_SVD = TruncatedSVD(n_components = 50)
    # X_train_data = scipy.sparse.csr_matrix(truncated_SVD.fit_transform(X_train_data))
    #
    # truncated_SVD = TruncatedSVD(n_components = 50)
    # X_validation_data = scipy.sparse.csr_matrix(truncated_SVD.fit_transform(X_validation_data))

    # train/learn the weights for the matrix W
    W = logisic_reg_train(X_train_data, X_train_classifications)

    # append a column of 1's to the validation data, this is adding an extra feature of all 1's per PDF spec and Piazza
    column_of_ones = np.full((X_validation.shape[0], 1), 1)
    X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, X_validation_data)), dtype = "float64")
    # TODO: Normalize the validation set using the same sums as the training set (Per Trilce)

    # same thing but use test data instead
    # X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, test_data)), dtype = "float64")

    # normalize the features (sum each column up and divide each nonzero element by that columns sum)
    X = normalize_columns(X)

    # will return the labels on the validation data, will also print our accuracy
    log_reg_predict(X, W, X_validation_classification, "validation")

    # labels = log_reg_predict(X, W, None, "testing")
    # if predicting on test
    # output_predictions("log_reg_testdata_output.csv", labels, 12001)

# logisic_reg_train: Logistic reg. implementation using Gradient Descent to find the matrix W
# that maximizes the probabilty we predict the correct class Y given features X
# This function is completely based on the PDF of project 2 under 'Log. Reg. implementation'
def logisic_reg_train(X_train, Y):

    # tunable parameters that will heavily impact the accuracy and convergence rate of Gradient Descent
    print("Shape of input: " + str(X_train.shape))
    learning_rate = 0.0001
    print("Learning rate: " + str(learning_rate))
    num_of_training_iterations = 1000
    print("Num of training iterations: " + str(num_of_training_iterations))
    lambda_regularization = .1
    print("Lambda regularization value: " + str(lambda_regularization))

    # num of examples
    m = X_train.shape[0]
    # num of classes
    k = 20
    # num of features
    n = X_train.shape[1]

    # (num_of_classes, num_of_examples) -> (m, k) matrix, where the entry delta,ij = 1 if for that example j the class is i
    delta = np.zeros((k, m))
    delta = scipy.sparse.csr_matrix(initialize_delta(delta, Y))

    # append column of 1s to sparse matrix X_train (per PDF and Piazza for something to do with normalization)
    column_of_ones = np.full((m, 1), 1)
    print(X_train.shape)
    print(type(X_train))

    X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, X_train)), dtype = np.float64)
    # normalize the features (sum each column up and divide each nonzero element by that columns sum)
    X = normalize_columns(X)

    # Weights for calculating conditional probability, initialized as all 0
    #W = scipy.sparse.csr_matrix(np.random.randn(k, n+1))
    W = scipy.sparse.csr_matrix(np.zeros((k, n+1), dtype=np.float64))
    # TODO: Make the weight matrix here random, then in for loop we have to normalize.

    for i in range(num_of_training_iterations):
        print("iteration" + str(i))
        # matrix of probabilities, P( Y | W, X) ~ exp(W * X^T)
        Z = (W.dot(X.transpose())).expm1()
        # Z.data = Z.data + 1
        # gradient w.r.t. Weights with regularization
        dZ = ((delta - Z) * X) - (lambda_regularization * W)
        # learning rule
        W = W + (learning_rate * dZ)

        # make predictions training data for each iteration, this adds min. time as the heaviest thing is normalizing
        #log_reg_predict(X, W, Y, "training")

    # return matrix of weights to use for predictions
    return W

# Set index equal to 1 if it's the same index as the class , 0 for all other classes, (Dirac Delta function)
# returns a matrix based on Dirac Delta function
def initialize_delta(delta, Y):
    Y_values = Y.data
    current_example = 0

    # go through each examples classification and index into the matrix delta and set that indice to 1
    # need to subtract 1 from the label because labels are 1-indexed
    for label in Y_values:
        # for class label on the current example, set index = 1
        delta[label-1, current_example] = 1
        current_example += 1

    return delta

# normalize_columns: takes the sum of every column and divides the nonzero data for a feature
# by that features summation
 # TODO: study python broadcasting...
def normalize_columns(Z):

    # take the sum of each column
    column_sums = np.array(Z.sum(axis=0))[0,:] # column vector
    row_indices, col_indices = Z.nonzero()
    Z.data /= column_sums[col_indices]  #TODO: this is wild

    return Z

# log_reg_predict: returns the predictions for the given data X. These predictions were
# learned by the weight matrix W which we trained using GD in logisic_reg_train
# Also prints the accuracy for the given data
def log_reg_predict(X, W, Y, predictions_on = "Training"):
    predictions = (W.dot(X.transpose())).expm1()
    predictions = predictions.toarray()

    max_value = -math.inf
    max_index = -1
    labels = []

    # for every example
    for j in range(predictions.shape[1]):
        for i in range(20):
            #print(str(i) + " : " + str(predictions[i][j]))
            if predictions[i][j] > max_value:
                max_value = predictions[i][j]
                max_index = i+1
        labels.append(max_index)
        # print("")
        max_value = -math.inf
        max_index = -1

    if Y != None:
        accuracy = 0
        for i in range(len(labels)):
            if labels[i] == Y[i]:
                accuracy += 1
        accuracy /= len(labels)
    print(accuracy)

    return labels

# output_predictions()
# Outputs the predictions from classification and outputs them into a file.
def output_predictions(file_name, predictions, starting_num):

    output_file = open(file_name, "w")

    output_file.write("id,class\n")

    i = 0
    for prediction in predictions:
        index = starting_num + i
        output_file.write("%d,%d\n" % (index, int(predictions[i])))
        i += 1

    output_file.close()

# load_classes()
# Loads the file that has the newsgroup classifications in it and returns
# the classifications as a list.
def load_classes(file_name):
    file = open(file_name, "r")

    classes = []

    for line in file:
        line = line.strip()
        split_line = line.split(" ")
        classes.append(split_line[1])

    return classes


if __name__ == "__main__":
    main()
