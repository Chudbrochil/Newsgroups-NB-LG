import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import scipy.sparse


# build_confusion_matrix()
# Builds the confusion matrix for either naive bayes' or logistic regression.
# Our goal is to have a strong diagonal which corresponds to good correlation
# between validation data classifications and our predictions.
def build_confusion_matrix(predictions, true_classes, classes, file_name):
    confusion_matrix = np.zeros(len(classes), len(classes), dtype=np.int64)
    len_pred = len(predictions)
    true_classes = true_classes.data
    # for every class prediction and true class value
    for i in range(len_pred):
        # we hope that these two are equal for a strong diagonal correlation
        confusion_matrix[predictions[i]-1, true_classes[i]-1] += 1

    confusion_matrix_df = pd.DataFrame(confusion_matrix, index= classes)

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

    confusion_matrix_df.to_csv(file_name, sep=",", header=classes)


# determine_most_important_features()
# This gave us the most important features np list file. This is no longer used as we
# have saved that file and read it in instead of it recomputing it on every run.
def determine_most_important_features():
    # the below code was used to generate the list of most important features rankings
    # if is_Nb:
    #     X_train = data[:, :-1]
    #     X_train_class = data[:, -1:].todense()
    #
    # idk = np.array(mutual_info_classif(X_train, X_train_class))
    # idk.dump("list_of_most_important_features_ranking.dat")
    amount_of_features = 60000
    rankings_of_each_feature = np.load("list_of_most_important_features_ranking.dat")
    most_important_features = np.argpartition(rankings_of_each_feature, -amount_of_features)[-amount_of_features:]
    # print(most_important_features)
    # match_variable_nums(most_important_features)
    return most_important_features


# match_variable_nums(): used to output top variables to a file
def match_variable_nums(int_total_prob):
    vocab = pd.read_csv('vocabulary.txt', sep=" ", header=None)
    vocab_values = vocab.values
    most_important_vocab = vocab_values[int_total_prob, 0]
    most_important_vocab = pd.DataFrame(most_important_vocab)
    most_important_vocab.to_csv("top_100_vocabulary.txt", header=None)


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
