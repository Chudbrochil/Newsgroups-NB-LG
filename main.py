import pandas as pd
import scipy.sparse
import numpy as np
import time

# indexing into sparse matrix:
# https://stackoverflow.com/questions/24665269/how-do-you-edit-cells-in-a-sparse-matrix-using-scipy

# https://stackoverflow.com/questions/38836100/accessing-sparse-matrix-elements

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

if __name__ == "__main__":
	main()
