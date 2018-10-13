import naive_bayes as nb
import logistic_regression as lr
import utilities as util
import scipy.sparse
from sklearn.model_selection import train_test_split
import argparse

"""
    Project 2 CS 529 - Naive Bayes and Logistic Regression from scratch

    @authors:
        Tristin Glunt | tglunt@unm.edu
        Anthony Galczak | agalczak@unm.edu

    Required Libraries:
        - SciPy 1.0.0 (loading the npz format as a csr_matrix) or higher
"""

def main():

    description_string = "Anthony Galczak/Tristin Glunt\n" \
                         "CS529 - Machine Learning\n" \
                         "Project 2 - Naive Bayes' & Logistic Regression\n" \
                         "NB & LR applied to \"Bag of Words\" document classification."

    parser = argparse.ArgumentParser(description=description_string, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', dest='is_tuning', action='store_true',
                        help="Set this to true if you would like to do tuning for the algorithm you specify.")
    parser.add_argument('-a', dest='algorithm', type=str, action='store', nargs='?',
                        default="nb", help='Which algorithm do you want to use. Acceptable options are \"nb\" or \"lr\".')
    args = parser.parse_args()

    # Selecting algorithm based on cli option
    use_naive_bayes = True
    if args.algorithm.lower() == "nb":
        print("Running Naive Bayes'.")
    elif args.algorithm.lower() == "lr":
        print("Running Logistic Regression.")
        use_naive_bayes = False
    else:
        print("No algorithm selected, defaulting to running Naive Bayes'.")

    # This is a list of hard-coded parameters. They could all be easily brought in
    # as command line arguments, but this is likely unnecessary.
    #betas = [.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1]
    betas = [.01] # Range of betas for tuning Naive Bayes'.
    beta = .01 # Beta variable for NB running on testing data.
    num_of_iterations = 50 # Number of weight updates in logistic regression
    learning_rate = .001 # Learning or eta term
    penalty_term = .001 # Penalty or lambda term
    show_matrix = False # Whether or not to show confusion matrix plot
    feature_selection = False # whether or not you would like to use feature selection

    # Lists of learning_rate and penalty_terms for tuning logistic regression.
    learning_rate_list = [.001, .0025, .0050, .0075, .01]
    penalty_term_list = [.001, .0025, .0050, .0075, .01]

    # Loads in a sparse matrix (csr_matrix) from a npz file.
    training_data = scipy.sparse.load_npz("training_sparse.npz")
    classes = util.load_classes("newsgrouplabels.txt")

    if args.is_tuning == True:
        print("Tuning mode on.")
        # Splits our data into training data and validation data.
        X_train, X_validation = train_test_split(training_data, test_size = .2, shuffle = True)

        if use_naive_bayes == True:
            # Tuning our naive bayes' given a range of Beta variables.
            nb.nb_tuning(X_train, X_validation, betas, show_matrix, classes, feature_selection)
        else:
            # Tuning Logistic Regression using a range of eta and lambda.
            lr.lr_tuning(X_train, X_validation, num_of_iterations, learning_rate_list, penalty_term_list, classes)
    else:
        # Loading the testing data fromW an npz file also.
        test_data = scipy.sparse.load_npz("testing_sparse.npz")

        if use_naive_bayes == True:
            # Run Naive Bayes' against the testing data, no validation dataset.
            nb.nb_solve(training_data, test_data, beta, classes, feature_selection)
        else:
            lr.lr_solve(training_data, test_data, learning_rate, penalty_term, num_of_iterations, classes)


if __name__ == "__main__":
    main()
