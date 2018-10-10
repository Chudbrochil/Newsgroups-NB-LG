import numpy as np

num_of_classes = 20 # TODO: Remove this, we can run load_classes()


# build_confusion_matrix()
# Builds the confusion matrix for either naive bayes' or logistic regression.
# Our goal is to have a strong diagonal which corresponds to good correlation
# between validation data classifications and our predictions.
def build_confusion_matrix(predictions, true_classes):
    confusion_matrix = np.zeros((num_of_classes, num_of_classes), dtype=np.int64)
    len_pred = len(predictions)
    true_classes = true_classes.data
    print(predictions[0])
    print(true_classes[0])
    # for every class prediction and true class value
    for i in range(len_pred):
        # we hope that these two are equal for a strong diagonal correlation
        confusion_matrix[predictions[i]-1, true_classes[i]-1] += 1

    print(confusion_matrix)
    return confusion_matrix


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
