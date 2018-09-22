import pandas as pd
import scipy.sparse
import numpy as np
import time

# indexing into sparse matrix:
# https://stackoverflow.com/questions/24665269/how-do-you-edit-cells-in-a-sparse-matrix-using-scipy

# https://stackoverflow.com/questions/38836100/accessing-sparse-matrix-elements

"""
    @authors:
    Required Libraries: SciPy 1.0.0 (loading the npz format as a csr_matrix)
"""


def main():
    #this should load in a csr_matrix
    data = scipy.sparse.load_npz("sparse_matrix_convert.npz")
    print("Shape of matrix: " + str(data.shape))
    # appears to be 0-indexed, and pandas chopped off the first row, so we need to start at the second row, and i didn't chop off first column
    print("Value at (row 1, col 12): " + str(data[0, 12]))
    print("Value that is a 0: " + str(data[0, 1]))
    print("Column M in Excel file: ")
    print(data[:, 11])
    print("First column in excel file:")
    print(data[:, 0])
    print("Entire data:")
    print(data.get_shape())
    print("Classifications " + str(data[:, -1:]))

    #start = time.time()
    # returns a tuple of lists ([row_indices], [col_indices])
    non_zero_data = data.nonzero()

    # How to loop through non zero data:
    print(non_zero_data)
    # print(len(non_zero_data[0]))
    #
    # print(len(non_zero_data[1]))
    start = time.time()
    # loop through every nonzero element
    for i in range(len(non_zero_data[0])):
        row_ele = non_zero_data[0][i]
        col_ele = non_zero_data[1][i]
        ran = data[row_ele, col_ele]
        # print("Row: " + str(row_ele) + " col: " + str(col_ele) + " Value: " + str(data[row_ele, col_ele]))

    end = time.time()
    total_time = end - start
    print(total_time)

    # get possible classifications
    classes = get_classifications(data[:, -1:])
    print("\n classes: " + str(classes))

    #calculate total words in each class
    #TODO write function to calculate total words in each class
    total_words_in_class = determine_total_words_in_classes(data, non_zero_data, classes)

    # pass only the column of classifications
    prior_probabilities= determine_prior_probabilities(data[:, -1:])

    # pass the dataset except the classifications
    likelihood_probabilities = determine_likelihoods(data, non_zero_data, classes, total_words_in_class)

# for every class, count the total amount of words in that class
def determine_total_words_in_classes(data, non_zero_data, classes):
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

# get_classifications()
# Obtaining the classifications from our data. For the DNA data, should be ["IE", "EI", "N"]
def get_classifications(class_list):
    classes = set()

    # have to convert the csr_vector to an array
    class_list = class_list.toarray()

    for list in class_list:
        classes.add(list[0])

    list_of_classes = []
    for element in classes:
        list_of_classes.append(element)

    return list_of_classes

if __name__ == "__main__":
    main()
