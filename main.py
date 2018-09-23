import pandas as pd
import scipy.sparse
import numpy as np
import math
import time
import copy

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
    data = scipy.sparse.load_npz("sparse_matrix_convert.npz")

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



# for every class, count the total amount of words in that class
def determine_total_words_in_classes(data):

    # We don't want the class counts to interfere with data counts
    classifications = data[:,-1:]
    data_without_classes = copy.deepcopy(data)
    data_without_classes = data_without_classes[:,0:-1]

    # Get the sum of each row, this returns a column vector
    row_sums = data_without_classes.sum(axis=1)

    # Initializing 20 dictionary elements for each newsgroup
    total_words_in_class = {}
    for x in range(1,21):
        total_words_in_class["class" + str(x)] = 0

    for x in range(12000):
        current_class = classifications.data[x]
        total_words_in_class["class" + str(current_class)] += row_sums[x][0]

    #print(total_words_in_class)
    return total_words_in_class


# return a dictionary of the prior probabilities for ["class_k"]
# calculate P(Y) -> # of examples labeled with class k / total examples

# TODO: possibly change hard coded iteration
def determine_prior_probabilities(classifications):

    class_counts = {}
    prior_probabilities = {}

    # initialize class counts for dictionary
    for i in range(1, 21):
        class_counts["class" + str(i)] = 0

    # add 1 for every label you encounter (1 instance)
    for label in classifications.data:
        class_counts["class" + str(label)] += 1

    # calculate the prior probabilities by dividing each class count by the total examples
    for i in range(1, 21):
        prior_probabilities["class" + str(i)] = class_counts["class" + str(i)] / len(classifications.data)
        #print("Prior probability for class " + str(i) + " " + str(prior_probabilities["class" + str(i)]))

    return prior_probabilities


# build a matrix: (classes, features) -> value is P(X|Y)
# return matrix of probabilites
# calculate P(X|Y) -> count # words in feature i with class k / total words in class k
def determine_likelihoods(data, non_zero_data, total_words_in_class):
    # num of features + 1
    laplace_denom = 61190

    # TODO: Re-write this comment a bit more concisely (Anthony)
    # Calculating row constants for initializing our 2D likelihood matrix.
    # Since our formula is P(X|Y) = ()(Count of X in Y) + 1/61190) / ()(words in Y) + 1)
    # We can split this up to say:
    # (((Count of X in Y) + 1/61190) / (Words in Y + 1)) +
    # ((1/61190) / (Words in Y + 1))
    # We are initializing our values to the second term...
    # ((1/61190) / (Words in Y + 1))
    # TODO: Verify that 61190 is our number for laplace smoothing
    initial_values = []
    for key, value in total_words_in_class.items():
        initial_value = 1.0 / (laplace_denom * int(value))
        initial_values.append(initial_value)

    # Initializing our matrix with the second term, we will add the first term
    # to these values in our "count of Xi in Yk" calculation.
    likelihood_matrix = np.zeros((20, 61189))
    for x in range(20):
        for y in range(61189):
            likelihood_matrix[x][y] = initial_values[x]

    length_of_nonzero_data = len(non_zero_data[0])

    # saving current row saves us ~1.5m hits for the entire data
    current_row_index = -1
    for i in range(length_of_nonzero_data):

        # getting coordinates of nonzero ele
        row_index = non_zero_data[0][i]
        col_index = non_zero_data[1][i]

        #if we're dealing with a new row
        if(row_index != current_row_index):
            current_classification = data[row_index, -1:].data[0]
            current_row_index = row_index

        current_val = data[row_index, col_index]

        # TODO: 0-index'ing vs 1-index'ing, hacky fix for now
        current_likelihood = likelihood_matrix[current_classification - 1][col_index]
        current_likelihood += (current_val / total_words_in_class["class" + str(current_classification)])

        # TODO: 0-index'ing vs 1-index'ing, hacky fix for now
        likelihood_matrix[current_classification - 1][col_index] = current_likelihood


    print(np.sum(likelihood_matrix, axis=0))
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
