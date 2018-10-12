import utilities as util
import math
import numpy as np
import pylab as p
import scipy.sparse
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata

# only global to avoid chain of returns
training_column_sums = np.array([])

# lr_solve()
# Trains logistic regression against some training data and then outputs predictions
# for some given testing data. Learning rate, penalty term (lambda) and num. of iterations
# are all tunable variables but they are brought in from main.
def lr_solve(training_data, test_data, learning_term, penalty_term, num_of_iterations):
    global training_column_sums

    training_data_no_classifications = training_data[:, :-1]
    training_data_classifications = training_data[:, -1:]

    W = lr_train(training_data_no_classifications, training_data_classifications, learning_term, penalty_term, num_of_iterations)

    column_of_ones = np.full((test_data.shape[0], 1), 1)
    # TODO: Normalize the validation set using the same sums as the training set (Per Trilce)
    X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, test_data)), dtype = "float64")

    row_indices, col_indices = X.nonzero()
    X.data /= training_column_sums[col_indices]

    predictions = lr_predict(X, W, None)

    util.output_predictions("log_reg_output.csv", predictions, training_data.shape[0] + 1)


# logistic_regression_solution: preprocessing and steps needed to use the logitic reg. alg
# Trains using Gradient descents
# TODO: Do tuning for eta(learning rate), lambda(penalty term), and 1 vs. 1000(10000?) iterations
def lr_tuning(X_train, X_validation):
    global training_column_sums
    # use feature selection by Naive Bayes likelihood matrix
    most_valuable_features = util.determine_most_important_features()

    X_train_classifications = X_train[:, -1:]
    X_train_data = X_train[:, most_valuable_features]

    X_validation_classification = X_validation[:, -1:]
    X_validation_data = X_validation[:, most_valuable_features]


    # Lists of values we are using for penalty term and learning rate
    # Using 5 terms each which gives 5x5 = 25 data points vs. accuracy.
    # Scaling between 1-10000 num of iterations adds another dimension as well.
    learning_rate_list = [0.0001, .001, .0025, .0050, .0075, .01]
    penalty_term_list = [.0001, .001, .0025, .0050, .0075, .01]
    #learning_rate_list = [.001, .005, .01]
    #penalty_term_list = [.001, .005, .01]
    #learning_rate_list = np.arange(.001, .01001, 5)
    #penalty_term_list = np.arange(.001, .01001, 5)

    accuracies = []
    for learning_rate in learning_rate_list:
        for penalty_term in penalty_term_list:

            # train/learn the weights for the matrix W
            W = lr_train(X_train_data, X_train_classifications, learning_rate, penalty_term, 100)

            # append a column of 1's to the validation data, this is adding an extra feature of all 1's per PDF spec and Piazza
            column_of_ones = np.full((X_validation.shape[0], 1), 1)
            X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, X_validation_data)), dtype = "float64")

            # normalize the features (sum each column up and divide each nonzero element by that columns sum)
            # after empircal tests, not normalizing the validation data has performed the best
            # X = normalize_columns(X)

            # will return the labels on the validation data, will also print our accuracy
            predictions = lr_predict(X, W, X_validation_classification)

            accuracy = 0
            for i in range(X_validation_data.shape[0]):
                if(predictions[i] == X_validation_classification[i]):
                    accuracy += 1
            accuracy /= X_validation_data.shape[0]
            print("Accuracy on validation set with learning_rate: %f and penalty term: %f, %f" % (learning_rate, penalty_term, accuracy))

            accuracies.append((learning_rate, penalty_term, accuracy))

            # TODO: Could put boolean flag for "build_confusion_matrix" here....
            # TODO: This is in a weird place. We don't need a confusion_matrix for each tuned variable (or do we?)
            classes = util.load_classes("newsgrouplabels.txt")
            util.build_confusion_matrix(predictions, X_validation_classification, classes, "log_reg_confusionMatrix.csv")


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = [i[0] for i in accuracies] # Gathering lambda values
    y = [i[1] for i in accuracies] # Gathering eta (learning_rate) values
    z = [i[2] for i in accuracies] # Gathering accuracy values

    # https://stackoverflow.com/questions/4363857/matplotlib-color-in-3d-plotting-from-an-x-y-z-data-set-without-using-contour
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))

    X, Y = np.meshgrid(xi, yi)
    Z = griddata(x, y, z, xi, yi, interp='linear')

    surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap=cm.jet,
                            linewidth=1, antialiased=True)

    ax.set_zlim3d(np.min(Z), np.max(Z))
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('Accuracy')
    fig.colorbar(surf)
    plt.show()

    print(accuracies)

    # labels = log_reg_predict(X, W, None, "testing")
    # if predicting on test
    # output_predictions("log_reg_testdata_output.csv", labels, 12001)

# lr_train: Logistic reg. implementation using Gradient Descent to find the matrix W
# that maximizes the probabilty we predict the correct class Y given features X
# This function is completely based on the PDF of project 2 under 'Log. Reg. implementation'
def lr_train(X_train, Y, learning_rate, penalty_term, num_of_iterations):

    # tunable parameters that will heavily impact the accuracy and convergence rate of Gradient Descent
    #learning_rate = 0.05 # .001 best
    print("Learning rate: " + str(learning_rate))
    #num_of_training_iterations = 1
    print("Num of iterations: " + str(num_of_iterations))
    #lambda_regularization = .01 # .1 best
    print("Lambda(penalty_term) value: " + str(penalty_term))

    # num of examples
    m = X_train.shape[0]
    # num of classes
    k = 20 # TODO: Hard coded variable (num_of_classes)
    # num of features
    n = X_train.shape[1]

    # (num_of_classes, num_of_examples) -> (m, k) matrix, where the entry delta,ij = 1 if for that example j the class is i
    delta = np.zeros((k, m))
    delta = scipy.sparse.csr_matrix(initialize_delta(delta, Y))

    # append column of 1s to sparse matrix X_train (per PDF and Piazza for something to do with normalization)
    column_of_ones = np.full((m, 1), 1)

    X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, X_train)), dtype = np.float64)
    # normalize the features (sum each column up and divide each nonzero element by that columns sum)
    X = normalize_columns(X)

    # Weights for calculating conditional probability, initialized as all 0
    #W = scipy.sparse.csr_matrix(np.random.randn(k, n+1))
    W = scipy.sparse.csr_matrix(np.zeros((k, n+1), dtype=np.float64))
    # TODO: Make the weight matrix here random, then in for loop we have to normalize.

    for i in range(num_of_iterations):
        print("iteration" + str(i))

        # matrix of probabilities, P( Y | W, X) ~ exp(W * X^T)
        Z = (W.dot(X.transpose())).expm1()
        Z = normalize_columns(Z)
        # gradient w.r.t. Weights with regularization
        dZ = ((delta - Z) * X) - (penalty_term * W)
        # learning rule
        W = W + (learning_rate * dZ)

        # make predictions training data for each iteration, this adds min. time as the heaviest thing is normalizing
        #log_reg_predict(X, W, Y, "training")

    # return matrix of weights to use for predictions
    return W

# Set index equal to 1 if it's the same index as the class , 0 for all other classes, (Dirac Delta function)
# returns a matrix based on Dirac Delta function
def initialize_delta(delta, Y):
    Y_values = Y.data
    current_example = 0

    # go through each examples classification and index into the matrix delta and set that indice to 1
    # need to subtract 1 from the label because labels are 1-indexed
    for label in Y_values:
        # for class label on the current example, set index = 1
        delta[label-1, current_example] = 1
        current_example += 1

    return delta

# normalize_columns: takes the sum of every column and divides the nonzero data for a feature
# by that features summation
 # TODO: study python broadcasting...
def normalize_columns(Z):
    global training_column_sums

    # take the sum of each column
    column_sums = np.array(Z.sum(axis=0))[0,:] # column vector
    row_indices, col_indices = Z.nonzero()
    # print(column_sums.shape)
    Z.data /= column_sums[col_indices]  #TODO: this is wild

    if len(training_column_sums) == 0:
        training_column_sums = column_sums

    return Z

# log_reg_predict: returns the predictions for the given data X. These predictions were
# learned by the weight matrix W which we trained using GD in logisic_reg_train
# Also prints the accuracy for the given data
def lr_predict(X, W, Y):
    predictions = (W.dot(X.transpose())).expm1()

    # take maximum and get index for every example
    maximum_index_for_each_example = predictions.argmax(axis=0).ravel().tolist()

    labels = []
    for i in range(predictions.shape[1]):
        labels.append(maximum_index_for_each_example[0][i] + 1)

    # print(labels)
    return labels



    # TODO: figure out dimensionality reduction techinique
    # truncated_SVD = TruncatedSVD(n_components = 50)
    # X_train_data = scipy.sparse.csr_matrix(truncated_SVD.fit_transform(X_train_data))
    #
    # truncated_SVD = TruncatedSVD(n_components = 50)
    # X_validation_data = scipy.sparse.csr_matrix(truncated_SVD.fit_transform(X_validation_data))
