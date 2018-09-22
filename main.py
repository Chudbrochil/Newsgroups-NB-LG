import pandas as pd
import scipy.sparse
import numpy as np
import time

# indexing into sparse matrix:
# https://stackoverflow.com/questions/24665269/how-do-you-edit-cells-in-a-sparse-matrix-using-scipy

# https://stackoverflow.com/questions/38836100/accessing-sparse-matrix-elements

"""
	@authors:
	Required Libraries: SciPy 1.0.0 (loading the npz format as a csr_matrix)
"""


def main():
	#this should load in a csr_matrix
	data = scipy.sparse.load_npz("sparse_matrix_convert.npz")
	print("Shape of matrix: " + str(data.shape))
	# appears to be 0-indexed, and pandas chopped off the first row, so we need to start at the second row, and i didn't chop off first column
	print("Value at (row 1, col 12): " + str(data[0, 12]))
	print("Value that is a 0: " + str(data[0, 1]))
	print("Column M in Excel file: ")
	print(data[:, 11])
	print("First column in excel file:")
	print(data[:, 0])
	print("Entire data:")
	print(data.get_shape())
	print("Classifications " + str(data[:, -1:]))

	start = time.time()
	# returns a tuple of lists ([row_indices], [col_indices])
	non_zero_data = data.nonzero()

	#loop through every nonzero element
	for rowEle in non_zero_data[0]:
		for colEle in non_zero_data[1]:
			# print("Row index: " + str(rowEle) + " Col index: " + str(colEle))
			# print(data[rowEle, colEle])
			randomAssignment = data[rowEle, colEle]
			break
		# break

	end = time.time()
	total_time = end - start
	print(total_time)

	# get possible classifications
	classes = get_classifications(data[:, -1:])

	#calculate total words in each class
	#TODO write function to calculate total words in each class

	# pass only the column of classifications
	prior_probabilities= determine_prior_probabilities(data[:, -1:])
	
	# pass the dataset except the classifications
	likelihood_probabilities = determine_likelihoods(data[:, :-1])

# return a list of probabilities corresponding to that classification
# [0.1, 0.05, ..., 0.8]
# [1,   2,  ...  , 14]
# calculate P(Y) -> # of examples labeled with class k / total examples
def determine_prior_probabilities(classificaitons):
	return ""

# build a matrix: (classes x features) -> value is P(X|Y)
# return matrix of probabilites
# calculate P(X|Y) -> count # words in feature i with class k / total words in class k
def determine_likelihoods(data, classes, total_words_in_class):
	likelihood_probabilities = []
	# for each classifcation
	for label in classes:

	return ""

# get_classifications()
# Obtaining the classifications from our data. For the DNA data, should be ["IE", "EI", "N"]
def get_classifications(class_list):
    classes = set()
    for list in class_list:
        classes.add(list[0])

    list_of_classes = []
    for element in classes:
        list_of_classes.append(element)

    return list_of_classes

if __name__ == "__main__":
	main()
