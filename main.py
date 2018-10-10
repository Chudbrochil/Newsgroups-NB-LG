import naive_bayes as nb
import logistic_regression as lr
import scipy.sparse
import copy
from sklearn.model_selection import train_test_split

"""
    Project 2 CS 529 - Naive Bayes and Logistic Regression from scratch

    @authors:
        Tristin Glunt | tglunt@unm.edu
        Anthony Galczak | agalczak@unm.edu

    Required Libraries:
        - SciPy 1.0.0 (loading the npz format as a csr_matrix) or higher
"""


def main():

    # Loads in a sparse matrix (csr_matrix) from a npz file.
    training_data = scipy.sparse.load_npz("training_sparse.npz")

    # TODO: TOP LEVEL VARIABLES, WILL BECOME CLI OPTIONS
    use_naive_bayes = True # False means using Logistic Regression
    is_tuning = False # False means we are running against testing data

    # Splits our data into training data and validation data.
    X_train, X_validation = train_test_split(training_data, test_size = .2, shuffle = True)

    # Loading the testing data from an npz file also.
    test_data = scipy.sparse.load_npz("testing_sparse.npz")

    if use_naive_bayes == True and is_tuning == True:
        # Tuning our naive bayes' given a range of Beta variables.
        betas = [.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1]
        nb.nb_tuning(X_train, X_validation, betas)
    elif use_naive_bayes == True and is_tuning == False:
        # Run Naive Bayes' against the testing data, no validation dataset.
        nb.nb_solve(training_data, test_data, .01)
    elif use_naive_bayes == False: # No tuning for logistic regression yet.
        lr.logistic_regression_solution(X_train, X_validation, test_data)


def determine_most_important_features(likelihood_probabilities):
    # take the sum of each column
    total_probabilities = likelihood_probabilities.sum(axis=0)
    # indices of top 1000 totals
    ind_total_prob = np.argpartition(total_probabilities, -60000)[-60000:]
    print(len(ind_total_prob))
    print(ind_total_prob)
    return ind_total_prob


if __name__ == "__main__":
    main()
