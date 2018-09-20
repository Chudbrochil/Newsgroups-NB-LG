import pandas as pd
import scipy.sparse
import numpy as np

def main():
	data = pd.read_csv("training.csv")
	print("convert to values...")
	data_values = data.values
	print("begin converting to csr_matrix...")
	matrix_converted = scipy.sparse.csr_matrix(data_values)
	# save the matrix to a file
	print(str(type(matrix_converted)))
	scipy.sparse.save_npz("sparse_matrix_convert.npz", matrix_converted)

if __name__ == "__main__":
	main()
