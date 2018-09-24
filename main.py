import pandas as pd
import scipy.sparse
import numpy as np
import math
import time
import copy
import operator

# indexing into sparse matrix:
# https://stackoverflow.com/questions/24665269/how-do-you-edit-cells-in-a-sparse-matrix-using-scipy

# https://stackoverflow.com/questions/38836100/accessing-sparse-matrix-elements

"""
    Project 2 CS 529 - Naive Bayes and Logistic Regression from scratch

    @authors:
        Tristin Glunt | tglunt@unm.edu
        Anthony Galczak | agalczak@unm.edu

    Required Libraries:
        - SciPy 1.0.0 (loading the npz format as a csr_matrix) or higher
"""


def main():
    #this should load in a csr_matrix
    data = scipy.sparse.load_npz("training_sparse.npz")

    # Training naive bayes
    np_train(data)


def np_train(data):
    # returns a tuple of lists that contain the non-zero indexes of the matrix data ([row_indices], [col_indices])
    non_zero_data = data.nonzero()

    # Loading in classes as strings from newsgrouplabels.txt
    classes = load_classes("newsgrouplabels.txt")

    # Calculate total # of words per a class. Needed for determine_likelihoods.
    total_words_in_class = determine_total_words_in_classes(data)

    # Calculate the ratio of prior probabilities, i.e. given_class/total_examples
    prior_probabilities= determine_prior_probabilities(data[:, -1:])

    # pass the dataset except the classifications
    likelihood_probabilities = determine_likelihoods(data, non_zero_data, total_words_in_class)

    # Making sure we can classify our training data correctly.
    #classify_training_data_test(data[:1, :-1], prior_probabilities, likelihood_probabilities)

    # Loading the testing data, getting our predictions, and then outputting them.
    test_data = scipy.sparse.load_npz("testing_sparse.npz")
    predictions = classify_training_data_test(test_data[:2, :], prior_probabilities, likelihood_probabilities)
    output_predictions("output.csv", predictions, 12001)


def output_predictions(file_name, predictions, starting_num):

    output_file = open(file_name, "w")

    i = 0
    for prediction in predictions:
        index = starting_num + i
        output_file.write("%d,%d\n" % (index, int(predictions[i])))
        i += 1

    output_file.close()


# y_prediction function
def classify_training_data_test(data, prior_probabilities, likelihood_probabilities):

    # calculate function for each classification
    sum_weighted_counts_likelihood = 0

    # Getting the length of the features, we should be passing in the data without
    # classifications. This should be equal to 61188
    length_of_features = data.shape[1]
    length_of_examples = data.shape[0]

    print("Classifying: %d examples, %d features." % (length_of_examples, length_of_features))

    predictions = {}

    # for every example
    for w in range(length_of_examples):
        print("On example: %d" % w)
        probabilities_for_classes = []

        # test every possible classification
        for i in range(20):
            log_prior = math.log(prior_probabilities["class" + str(i)])

            # go through every feature
            for j in range(length_of_features):
                # count for current feature for current example
                current_count = data[w, j]

                # log likelihood for current class and feature
                log_of_likelihood = math.log(likelihood_probabilities[i][j])

                multiplied_count_likelihood = current_count * log_of_likelihood

                sum_weighted_counts_likelihood += multiplied_count_likelihood

            probability_for_current_class = log_prior + sum_weighted_counts_likelihood
            sum_weighted_counts_likelihood = 0

            probabilities_for_classes.append(probability_for_current_class)

        example_prediction = max(enumerate(probabilities_for_classes), key=operator.itemgetter(1))[0] + 1 # NOTE: Since the classes are 1-indexed.

        predictions[w] = example_prediction


    print("Dictionary of predictions")
    print(predictions)
    return predictions


#def classify():


# for every class, count the total amount of words in that class
def determine_total_words_in_classes(data):

    # We don't want the class counts to interfere with data counts
    classifications = data[:,-1:]
    data_without_classes = data[:,0:-1]

    # Get the sum of each row, this returns a column vector
    row_sums = data_without_classes.sum(axis=1)

    # Initializing 20 dictionary elements for each newsgroup
    total_words_in_class = {}
    for x in range(20):
        total_words_in_class["class" + str(x)] = 0

    for x in range(12000):
        current_class = (classifications.data[x] - 1) # NOTE: The classifications are 1-index'ed. This is "the fix"
        total_words_in_class["class" + str(current_class)] += row_sums[x][0]

    return total_words_in_class


# return a dictionary of the prior probabilities for ["class_k"]
# calculate P(Y) -> # of examples labeled with class k / total examples
# TODO: possibly change hard coded iteration
def determine_prior_probabilities(classifications):

    class_counts = {}
    prior_probabilities = {}

    # initialize class counts for dictionary
    for i in range(20):
        class_counts["class" + str(i)] = 0

    # add 1 for every label you encounter (1 instance)
    for label in classifications.data:
        class_counts["class" + str(label - 1)] += 1 # NOTE: The classifications are 1-index'ed. This is "the fix"

    # calculate the prior probabilities by dividing each class count by the total examples
    for i in range(20):
        prior_probabilities["class" + str(i)] = class_counts["class" + str(i)] / len(classifications.data)

    return prior_probabilities


# build a matrix: (classes, features) -> value is P(X|Y)
# return matrix of probabilites
# calculate P(X|Y) -> count # words in feature i with class k / total words in class k
def determine_likelihoods(data, non_zero_data, total_words_in_class):

    likelihood_matrix = np.zeros((20, 61189)) # NOTE: 61189 because we have classifications also.
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
    for x in range(20):
        total_words = total_words_in_class["class" + str(x)]
        for y in range(61189):
            enhanced_likelihood = likelihood_matrix[x][y]
            enhanced_likelihood += (1.0 / 61188)
            enhanced_likelihood /= (total_words + 1)
            likelihood_matrix[x][y] = enhanced_likelihood

    return likelihood_matrix


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
