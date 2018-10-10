import utilities as util
import math
from sklearn.decomposition import TruncatedSVD

# logistic_regression_solution: preprocessing and steps needed to use the logitic reg. alg
# Trains using Gradient descent
def logistic_regression_solution(X_train, X_validation, test_data):
    likelihood_probabilities, prior_probabilities = nb_train(X_train, 0.001)
    array_of_most_imp_features = determine_most_important_features(likelihood_probabilities)

    spliced_X_train = scipy.sparse.csr_matrix(X_train.tocsc()[:, array_of_most_imp_features])
    spliced_X_validation= scipy.sparse.csr_matrix(X_validation.tocsc()[:, array_of_most_imp_features])
    print(spliced_X_train.shape)

    # separate features and classifications
    X_train_data = spliced_X_train[:, :-1]
    X_train_classifications = X_train[:, -1:]

    X_validation_data = spliced_X_validation[:, :-1]
    X_validation_classification = X_validation[:, -1:]

    # TODO: figure out dimensionality reduction techinique
    # truncated_SVD = TruncatedSVD(n_components = 50)
    # X_train_data = scipy.sparse.csr_matrix(truncated_SVD.fit_transform(X_train_data))
    #
    # truncated_SVD = TruncatedSVD(n_components = 50)
    # X_validation_data = scipy.sparse.csr_matrix(truncated_SVD.fit_transform(X_validation_data))

    # train/learn the weights for the matrix W
    W = logisic_reg_train(X_train_data, X_train_classifications)

    # append a column of 1's to the validation data, this is adding an extra feature of all 1's per PDF spec and Piazza
    column_of_ones = np.full((X_validation.shape[0], 1), 1)
    X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, X_validation_data)), dtype = "float64")
    # TODO: Normalize the validation set using the same sums as the training set (Per Trilce)

    # same thing but use test data instead
    # X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, test_data)), dtype = "float64")

    # normalize the features (sum each column up and divide each nonzero element by that columns sum)
    X = normalize_columns(X)

    # will return the labels on the validation data, will also print our accuracy
    predictions = log_reg_predict(X, W, X_validation_classification, "validation")

    confusion_matrix = build_confusion_matrix(predictions, X_validation_classification)
    np.savetxt("logisticregression_confusion_matrix.csv", confusion_matrix, delimiter=",", fmt='%10.5f')

    # labels = log_reg_predict(X, W, None, "testing")
    # if predicting on test
    # output_predictions("log_reg_testdata_output.csv", labels, 12001)

# logisic_reg_train: Logistic reg. implementation using Gradient Descent to find the matrix W
# that maximizes the probabilty we predict the correct class Y given features X
# This function is completely based on the PDF of project 2 under 'Log. Reg. implementation'
def logisic_reg_train(X_train, Y):

    # tunable parameters that will heavily impact the accuracy and convergence rate of Gradient Descent
    print("Shape of input: " + str(X_train.shape))
    learning_rate = 0.0001
    print("Learning rate: " + str(learning_rate))
    num_of_training_iterations = 1
    print("Num of training iterations: " + str(num_of_training_iterations))
    lambda_regularization = .1
    print("Lambda regularization value: " + str(lambda_regularization))

    # num of examples
    m = X_train.shape[0]
    # num of classes
    k = 20
    # num of features
    n = X_train.shape[1]

    # (num_of_classes, num_of_examples) -> (m, k) matrix, where the entry delta,ij = 1 if for that example j the class is i
    delta = np.zeros((k, m))
    delta = scipy.sparse.csr_matrix(initialize_delta(delta, Y))

    # append column of 1s to sparse matrix X_train (per PDF and Piazza for something to do with normalization)
    column_of_ones = np.full((m, 1), 1)
    print(X_train.shape)
    print(type(X_train))

    X = scipy.sparse.csr_matrix(scipy.sparse.hstack((column_of_ones, X_train)), dtype = np.float64)
    # normalize the features (sum each column up and divide each nonzero element by that columns sum)
    X = normalize_columns(X)

    # Weights for calculating conditional probability, initialized as all 0
    #W = scipy.sparse.csr_matrix(np.random.randn(k, n+1))
    W = scipy.sparse.csr_matrix(np.zeros((k, n+1), dtype=np.float64))
    # TODO: Make the weight matrix here random, then in for loop we have to normalize.

    for i in range(num_of_training_iterations):
        print("iteration" + str(i))
        # matrix of probabilities, P( Y | W, X) ~ exp(W * X^T)
        Z = (W.dot(X.transpose())).expm1()
        # Z.data = Z.data + 1
        # gradient w.r.t. Weights with regularization
        dZ = ((delta - Z) * X) - (lambda_regularization * W)
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

    # take the sum of each column
    column_sums = np.array(Z.sum(axis=0))[0,:] # column vector
    row_indices, col_indices = Z.nonzero()
    Z.data /= column_sums[col_indices]  #TODO: this is wild

    return Z

# log_reg_predict: returns the predictions for the given data X. These predictions were
# learned by the weight matrix W which we trained using GD in logisic_reg_train
# Also prints the accuracy for the given data
def log_reg_predict(X, W, Y, predictions_on = "Training"):
    predictions = (W.dot(X.transpose())).expm1()
    predictions = predictions.toarray()

    max_value = -math.inf
    max_index = -1
    labels = []

    # for every example
    for j in range(predictions.shape[1]):
        for i in range(20):
            #print(str(i) + " : " + str(predictions[i][j]))
            if predictions[i][j] > max_value:
                max_value = predictions[i][j]
                max_index = i+1
        labels.append(max_index)
        # print("")
        max_value = -math.inf
        max_index = -1

    if Y != None:
        accuracy = 0
        for i in range(len(labels)):
            if labels[i] == Y[i]:
                accuracy += 1
        accuracy /= len(labels)
    print(accuracy)

    return labels