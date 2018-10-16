import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse


# build_confusion_matrix()
# Builds the confusion matrix for either naive bayes' or logistic regression.
# Our goal is to have a strong diagonal which corresponds to good correlation
# between validation data classifications and our predictions.
def build_confusion_matrix(predictions, true_classes, classes, file_name, show_matrix):
    confusion_matrix = np.zeros(len(classes), len(classes), dtype=np.int64)
    len_pred = len(predictions)
    true_classes = true_classes.data
    # for every class prediction and true class value
    for i in range(len_pred):
        # we hope that these two are equal for a strong diagonal correlation
        confusion_matrix[predictions[i]-1, true_classes[i]-1] += 1

    confusion_matrix_df = pd.DataFrame(confusion_matrix, index= classes)

    if show_matrix == True:
        plt.imshow(confusion_matrix_df.values, cmap='viridis', interpolation='nearest')
        plt.xticks(np.arange(20), classes, rotation='85')
        plt.yticks(np.arange(20), classes)
        plt.tick_params(axis='both', labelsize='6')
        plt.xlabel("True classifications")
        plt.ylabel("Predicted classifications")
        plt.title("Confusion Matrix of Pred. Classes vs True Classes for Logistic Regression")
        plt.tight_layout()
        for (j, i), label in np.ndenumerate(confusion_matrix):
            if label != 0:
                plt.text(i,j,label,ha='center',va='center', size='6')
        plt.show()


    # # confusion_matrix_df.set_index(classes)
    confusion_matrix_df.to_csv(file_name, sep=",", header=classes)

def determine_most_important_features():
    print("Using feature selection.")
    amount_of_features_keeping = 60000
    likelihood_probabilities = np.load("likelihood_matrix.dat")
    likelihood_probabilities = likelihood_probabilities[:, :-1]
    # take the sum of each column
    total_probabilities = likelihood_probabilities.max(axis=0)

    # sort in descending order
    # total_probabilities = sorted(total_probabilities, reverse=True)
    # ind_total_prob = range(100)

    # take X amount of top probabilities, in no particular order
    ind_total_prob = np.argpartition(total_probabilities, -amount_of_features_keeping)[-amount_of_features_keeping:]
    #filtered_ind_total_prob = filter_based_on_counts(ind_total_prob)
    # match_variable_nums(ind_total_prob)
    return ind_total_prob

# filter_based_on_counts: a complicated filtering to help with feature seleciton that had
# little to no improvement
def filter_based_on_counts(ind_total_prob):
    training_data = scipy.sparse.load_npz("training_sparse.npz")
    training_data = training_data[:, :-1]
    # get data based on highest probabilities
    training_data = training_data[:, ind_total_prob]

    total_counts_each_feature =  np.array(training_data.sum(axis=0))[0,:]
    print(total_counts_each_feature)

    indices_and_counts = zip(ind_total_prob, total_counts_each_feature)

    sorted_indices_and_counts = sorted(indices_and_counts, key=lambda tup: tup[1])
    print("Sorted by counts" + str(sorted_indices_and_counts))

    choose_middle_100_words = sorted_indices_and_counts[:4900]
    # get first element of each tuple and make a list
    indices_of_choosen_words = [i[0] for i in choose_middle_100_words]

    print("Indices of middle 100 words: " + str(indices_of_choosen_words))

    return indices_of_choosen_words


# match_variable_nums(): used to output top variables to a file
def match_variable_nums(int_total_prob):
    vocab = pd.read_csv('vocabulary.txt', sep=" ", header=None)
    vocab_values = vocab.values
    # print("Shape of vocabulary " + str(vocab_values.shape))
    most_important_vocab = vocab_values[int_total_prob, 0]
    most_important_vocab = pd.DataFrame(most_important_vocab)
    most_important_vocab.to_csv("top_100_vocabulary.txt", header=None)
    # print("Shape of most important vocab" + str(most_important_vocab.shape))
    # print(most_important_vocab)

# output_predictions()
# Outputs the predictions from classification and outputs them into a file.
def output_predictions(file_name, predictions, starting_num):

    output_file = open(file_name, "w")
    print("Writing to file: %s" % file_name)
    output_file.write("id,class\n")

    i = 0
    for prediction in predictions:
        index = starting_num + i
        output_file.write("%d,%d\n" % (index, int(predictions[i])))
        i += 1

    output_file.close()


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
