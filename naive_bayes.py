import utilities as util
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


num_of_classes = 20 # TODO: Remove this global

# nb_solve()
# Training our naive bayes' algorithm against our full set of training data and
# then getting predictions on testing data and then outputting that to a file.
def nb_solve(training_data, testing_data, beta):
    likelihood_probabilities, prior_probabilities = nb_train(training_data, beta)
    predictions = nb_predict(testing_data, prior_probabilities, likelihood_probabilities, True)
    util.output_predictions("testing_predictions.csv", predictions, training_data.shape[0] + 1)


# nb_tuning()
# Tunes naive bayes for a range of Beta values. This method will run the Naive Bayes'
# algorithm for each of these Beta variables and then plot accuracy vs. the validation
# data set when it is done running.
def nb_tuning(X_train, X_validation, betas):
    print("Training set size: " + str(X_train.shape))
    print("Validation set size: " + str(X_validation.shape))
    classes = util.load_classes("newsgrouplabels.txt")

    X_validation_classification = X_validation[:, -1:]
    accuracies = []

    # Go through and train on each beta
    # TODO: Store likelihood_probabilities and prior_probabilities that best
    # classify the validation data and then use this to train
    for beta in betas:
        # Training naive bayes
        likelihood_probabilities, prior_probabilities = nb_train(X_train, beta)
        predictions = nb_predict(X_validation, prior_probabilities, likelihood_probabilities)

        util.build_confusion_matrix(predictions, X_validation_classification, classes, "naive_bayes_confusionMatrix.csv")

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

    util.output_predictions("validation_output.csv", predictions, X_train.shape[0])

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

# nb_train()
# Meta method for building P(Y) and P(X|Y) probabilities from Naive Bayes.
# This method will bring in a set of data (training data, typically separated from
# validation data), and a Beta tuning variable.
def nb_train(data, beta):
    # returns a tuple of lists that contain the non-zero indexes of the matrix data ([row_indices], [col_indices])
    non_zero_data = data.nonzero()

    # Loading in classes as strings from newsgrouplabels.txt
    classes = util.load_classes("newsgrouplabels.txt")

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
def nb_predict(data, prior_probabilities, likelihood_probabilities, is_testing = False):

    log_priors = []
    for value in prior_probabilities.values():
        log_value = math.log(value)
        log_priors.append(log_value)

    likelihood_probabilities.data = np.log(likelihood_probabilities.data)

    if is_testing == True:
        # TODO: Chop off the last column, Why do we have to do this?
        likelihood_probabilities = np.delete(likelihood_probabilities, -1, axis=1)

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

    # save likelihood matrix using numpy pickle
    likelihood_matrix.dump("likelihood_matrix.dat")
    return likelihood_matrix
