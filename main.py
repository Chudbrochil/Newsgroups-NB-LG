import pandas as pd
import scipy.sparse
import numpy as np
import time

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
    print("Shape of matrix: " + str(data.shape))
    print("Value at (row 0, col 12): " + str(data[0, 11]))
    print("Value that is a 0: " + str(data[0, 1]))
    print("Entire data shape: " + str(data.get_shape()))

    # returns a tuple of lists that contain the non-zero indexes of the matrix data ([row_indices], [col_indices])
    non_zero_data = data.nonzero()
    print(non_zero_data)

    # """ EXAMPLE OF RUNNING THROUGH ENTIRE MATIRX, O(n) """
    # start = time.time()
    # # loop through every nonzero element
    # for i in range(len(non_zero_data[0])):
    #     row_ele = non_zero_data[0][i]
    #     col_ele = non_zero_data[1][i]
    #     ran = data[row_ele, col_ele]
    #     # print("Row: " + str(row_ele) + " col: " + str(col_ele) + " Value: " + str(data[row_ele, col_ele]))
    #
    # end = time.time()
    # total_time = end - start
    # print(total_time)
    # """  --------------------------------------------  """

    # Loading in classes as strings from newsgrouplabels.txt
    classes = load_classes("newsgrouplabels.txt")

    # Calculate total # of words per a class. Needed for determine_likelihoods.
    total_words_in_class = determine_total_words_in_classes(data)

    # Calculate the ratio of prior probabilities, i.e. given_class/total_examples
    prior_probabilities= determine_prior_probabilities(data[:, -1:])

    # pass the dataset except the classifications
    likelihood_probabilities = determine_likelihoods(data, non_zero_data, classes, total_words_in_class)


# for every class, count the total amount of words in that class
def determine_total_words_in_classes(data):

    # We don't want the class counts to interfere with data counts
    classifications = data[:,-1:]
    data = data[:,0:-1]

    # Get the sum of each row, this returns a column vector
    row_sums = data.sum(axis=1)

    # Initializing 20 dictionary elements for each newsgroup
    total_words_in_class = {}
    for x in range(1,21):
        total_words_in_class["class" + str(x)] = 0

    for x in range(12000):
        current_class = classifications.data[x]
        total_words_in_class["class" + str(current_class)] += row_sums[x][0]

    print(total_words_in_class)
    return total_words_in_class


# return a dictionary of the prior probabilities for ["class_k"]
# calculate P(Y) -> # of examples labeled with class k / total examples

# TODO: possibly change hard coded iteration
def determine_prior_probabilities(classifications):

    class_counts = {}
    prior_probabilities = {}

    print(classifications.data)

    # initialize class counts for dictionary
    for i in range(1, 21):
        class_counts["class" + str(i)] = 0

    # add 1 for every label you encounter (1 instance)
    for label in classifications.data:
        class_counts["class" + str(label)] += 1

    # calculate the prior probabilities by dividing each class count by the total examples
    for i in range(1, 21):
        prior_probabilities["class" + str(i)] = class_counts["class" + str(i)] / len(classifications.data)
        print("Prior probability for class " + str(i) + " " + str(prior_probabilities["class" + str(i)]))

    return prior_probabilities




# build a matrix: (classes, features) -> value is P(X|Y)
# return matrix of probabilites
# calculate P(X|Y) -> count # words in feature i with class k / total words in class k
def determine_likelihoods(data, non_zero_data, classes, total_words_in_class):
    likelihood_probabilities = np.zeros((data.get_shape()[0], data.get_shape()[1]))
    print("Shape of likelihood matrix: " + str(likelihood_probabilities.shape))
    rows_nonzero = non_zero_data[0]
    cols_nonzero = non_zero_data[1]
    print("shape of rows_nonzero" + str(rows_nonzero.shape))
    print("Total nonzero examples: " + str(len(rows_nonzero)))

    # for each classifcation
    for label in classes:
        # for each feature
        for col in cols_nonzero:
            # add up words for every example
            for row in rows_nonzero:
                """ Calclate total words in current col with label """
                # if (data[row, :-1]) == label
                # temp_total += data[row, col]
            # temp_probability /= total_words_in_class[label]
            # likelihood_probabilities[label, col] = temp_probability

    return ""


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
