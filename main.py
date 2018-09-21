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
	print(data)

if __name__ == "__main__":
	main()
