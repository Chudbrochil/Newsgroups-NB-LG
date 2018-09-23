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
        Anthony Galczak | ...

    Required Libraries:
        - SciPy 1.0.0 (loading the npz format as a csr_matrix) or hihgher
"""


def main():
    #this should load in a csr_matrix
    data = scipy.sparse.load_npz("sparse_matrix_convert.npz")
    print("Shape of matrix: " + str(data.shape))
    print("Value at (row 0, col 12): " + str(data[0, 12]))
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

    #TODO load in file and get list of classifications... (can just hard code 0:20)
    classes = load_classes("newsgrouplabels.txt")
    #classifcation_file = ...

    #calculate total words in each class
    #TODO write function to calculate total words in each class
    total_words_in_class = determine_total_words_in_classes(data, non_zero_data, classes)

    # pass only the column of classifications
    #prior_probabilities= determine_prior_probabilities(data[:, -1:])

    # pass the dataset except the classifications
    #likelihood_probabilities = determine_likelihoods(data, non_zero_data, classes, total_words_in_class)




# for every class, count the total amount of words in that class
def determine_total_words_in_classes(data, non_zero_data, classes):

    # We don't want the class counts to interfere with data counts
    data = data[:,0:-1]

    # Get the sum of each row, this returns a column vector
    row_sums = data.sum(axis=1)

    total_words_in_class = {}

    #for x in range(12000):
    #    total_words_in_class["class" + str()]



    print(row_sums)
    print(len(row_sums))

    # Flattenning the list of lists and removing the
    #for list_element in row_sums:


    #for x in range(12000):
    #    print(columns[x][0])

    #print("data")
    #print(data)
    return ""



# return a list of probabilities corresponding to that classification
# [0.1, 0.05, ..., 0.8]
# [1,   2,  ...  , 14]
# calculate P(Y) -> # of examples labeled with class k / total examples
def determine_prior_probabilities(classifictions):
    return ""




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
