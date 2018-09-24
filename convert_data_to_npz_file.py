import pandas as pd
import scipy.sparse
import numpy as np

def main():
	data = pd.read_csv("testing.csv", header=None)
	print("convert to values...")
	data_values = data.values
	data_values = data_values[:, 1:]
	print("begin converting to csr_matrix...")
	matrix_converted = scipy.sparse.csr_matrix(data_values)
	# save the matrix to a file
	print(str(type(matrix_converted)))
	scipy.sparse.save_npz("testing_sparse.npz", matrix_converted)

if __name__ == "__main__":
	main()
