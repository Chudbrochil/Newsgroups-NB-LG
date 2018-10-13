import naive_bayes as nb
import logistic_regression as lr
import scipy.sparse
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
    use_naive_bayes = False # False means using Logistic Regression
    is_tuning = True # False means we are running against testing data

    # Splits our data into training data and validation data.
    X_train, X_validation = train_test_split(training_data, test_size = .2, shuffle = True)

    # Loading the testing data from an npz file also.
    test_data = scipy.sparse.load_npz("testing_sparse.npz")

    if use_naive_bayes == True and is_tuning == True:
        # Tuning our naive bayes' given a range of Beta variables.
        # betas = [.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1]
        betas = [.01]
        nb.nb_tuning(X_train, X_validation, betas)
    elif use_naive_bayes == True and is_tuning == False:
        # Run Naive Bayes' against the testing data, no validation dataset.
        nb.nb_solve(training_data, test_data, .01)
    elif use_naive_bayes == False and is_tuning == True:
        # Tuning Logistic Regression using a range of eta and lambda.
<<<<<<< HEAD
        lr.lr_tuning(X_train, X_validation, 1)
=======

        lr.lr_tuning(X_train, X_validation)
>>>>>>> 22cb8d81f39cec1f940340a4df81d07dad16a92f
    elif use_naive_bayes == False and is_tuning == False:
        lr.lr_solve(training_data, test_data, .001, .001, 10)


if __name__ == "__main__":
    main()
