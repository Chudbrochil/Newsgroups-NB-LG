import pandas as pd
import scipy.sparse
import numpy as np

# indexing into sparse matrix:
# https://stackoverflow.com/questions/24665269/how-do-you-edit-cells-in-a-sparse-matrix-using-scipy

def main():
	#this should load in a csr_matrix
	data = scipy.sparse.load_npz("sparse_matrix_convert.npz")
	print("Value at (row 25, col 15): " + str(data[25, 15]))
	print("test")

if __name__ == "__main__":
	main()
