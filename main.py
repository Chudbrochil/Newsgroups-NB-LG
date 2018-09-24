

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

num_of_classes = 20


def main():
    #this should load in a csr_matrix
    data = scipy.sparse.load_npz("training_sparse.npz")

    # Training naive bayes
    nb_train(data)


def nb_train(data):
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
    #classify_data(data[:1, :-1], prior_probabilities, likelihood_probabilities)

    # Loading the testing data, getting our predictions, and then outputting them.
    test_data = scipy.sparse.load_npz("testing_sparse.npz")

    predictions = nb_predict(data[:10, :-1], prior_probabilities, likelihood_probabilities)
    output_predictions("output.csv", predictions, 12001)


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


# nb_predict()
# Calculates the prediction function for Naive Bayes
# Classifies a set of data (validation or testing) based upon the likelihood
# matrix (P(X|Y)) and priors (P(Y)) that we calculated earlier.
def nb_predict(data, prior_probabilities, likelihood_probabilities):
    # Getting the length of the features, we should be passing in the data without
    # classifications. This should be equal to 61188
    length_of_features = data.shape[1]
    length_of_examples = data.shape[0]

    start_time = time.time()

    #gives a tuple of rows and columns of nonzero data
    nonzero_test_data = data.nonzero()

    print("Classifying: %d examples, %d features." % (length_of_examples, length_of_features))

    length_of_nonzero_test_data = len(nonzero_test_data[0])

    predictions = {}
    current_row_test_index = -1
    prev_row_index = -1
    new_starting_value_for_nonzero_matrix = 0

    # for every example (w is an index, but we're going to treat it like we're looping through every row at a time in data)
    for row in range(length_of_nonzero_test_data):
        current_nonzero_row_index = nonzero_test_data[0][row]

        # if we're dealing with a new example, we will calculate a prediction,
        # otherwise we will continue until we get to a new prediction. This gives us a way
        # to check our indexing for the tuples of indexing. We should always match the two.
        if(prev_row_index != current_nonzero_row_index):

            print("On example: %d" % current_nonzero_row_index)
            highest_prob = -math.inf
            highest_prob_index = -1

            # update last row index to the new row we're on
            prev_row_index = current_nonzero_row_index

            # test every possible classification for the current example (row)
            for k in range(20):
                # the number of entries in the current row (we will skip this amount to start at the next row)
                num_of_items_in_current_row = 0
                # summation variable in y_prediction function
                sum_weighted_counts_likelihood = 0

                log_prior = math.log(prior_probabilities["class" + str(k)])

                # loop through every nonzero feature (can skip 0 features because it would just add 0)
                for i in range(length_of_nonzero_test_data):

                    #TODO why do I have to catch this error? Investigate new_starting_value_for_nonzero_matrix
                    if((i + new_starting_value_for_nonzero_matrix) >= length_of_nonzero_test_data):
                        print("Catching possible index out of bounds exception")
                        break

                    # have to add new starting value because we are not remove elements that we've
                    # already visited in the tuple of nonzero indices
                    current_row = nonzero_test_data[0][i + new_starting_value_for_nonzero_matrix]

                    # if we're not on the same example, we need to break, go to next class,
                    # and repeat the same iterations but for the new class
                    if current_row != current_nonzero_row_index:
                        break

                    current_col = nonzero_test_data[1][i + new_starting_value_for_nonzero_matrix]
                    # weight of this feature in the new dataset
                    current_count = data[current_row, current_col]
                    # get the likelihood of current class for current column
                    log_of_likelihood = math.log(likelihood_probabilities[k][current_col])
                    multiplied_count_likelihood = current_count * log_of_likelihood
                    sum_weighted_counts_likelihood += multiplied_count_likelihood

                    # increment the total amount of features processed in this row ()
                    num_of_items_in_current_row += 1

                probability_for_current_class = log_prior + sum_weighted_counts_likelihood

                # Finding the class with highest probability prediction
                if probability_for_current_class > highest_prob:
                    highest_prob = probability_for_current_class
                    highest_prob_index = k

            #print("Num of iterations done: " + str(num_of_items_in_current_row))
            predictions[current_nonzero_row_index] = highest_prob_index + 1 # NOTE: Since the classes are 1-indexed.

            # after every classification has been processed, we need to update the starting point
            # to the next row of nonzero data
            new_starting_value_for_nonzero_matrix += num_of_items_in_current_row

    print("Dictionary of predictions")
    print(predictions)
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))
    return predictions


# determine_total_words_in_classes()
# Calculating how many total words are in each classification.
# This is useful in likelihood calculation in denominator.
def determine_total_words_in_classes(data):

    # We don't want the class counts to interfere with data counts
    classifications = data[:,-1:]
    data_without_classes = data[:,0:-1]

    # Get the sum of each row, this returns a column vector
    row_sums = data_without_classes.sum(axis=1)

    # Initializing 20 dictionary elements for each newsgroup
    total_words_in_class = {}
    for x in range(num_of_classes):
        total_words_in_class["class" + str(x)] = 0

    for x in range(12000):
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
def determine_likelihoods(data, non_zero_data, total_words_in_class):

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
            enhanced_likelihood += 1.0 / 100 #(1.0 / 61188)
            enhanced_likelihood /= (total_words + 61188 / 100)#(total_words + 1)
            likelihood_matrix[x][y] = enhanced_likelihood

    return likelihood_matrix


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
