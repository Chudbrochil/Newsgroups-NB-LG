import pandas as pd
import scipy.sparse
import numpy as np

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

	"""
		Possible way for us to loop through nonzero data only...not sure if this
	 	is more expensive than just looping through all data and ignoring zero values?
	"""
	# loop through every column
	for i in range(data.get_shape()[1]):
		# grab column i
		current_col_csr = data.getcol(i)
		# filter column i to only contain the nonzero entries
		nonzero_current_col = current_col_csr.nonzero()
		# says column entries are all 0s but they're actually i value
		# print non_zero values for column i and row_val
		for row_val in nonzero_current_col[0]:
			print("row: " + str(row_val) + " col: " + str(i) + " value: ", end="")
			print(data[row_val, i])



#may need to use data.nonzero() to get tuple of rows and column indices that aren't 0 so we know what to loop over

if __name__ == "__main__":
	main()
