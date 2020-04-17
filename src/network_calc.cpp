#include "network_calc.h"

namespace network_calc
{
	Eigen::MatrixXd multiply_matrices(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2)
	{
		// TODO: Consider changing return MatrixXd to VectorXd- you should always get an n*1 matrix back for a layer
		if (matrix1.cols() != matrix2.rows())
		{
			throw std::runtime_error("Cannot perform dot product, dimensions are unequal");
		}

		Eigen::MatrixXd output(matrix1.rows(), matrix2.cols());

		for (size_t i = 0; i < matrix1.rows(); i++)
		{
			for (size_t j = 0; j < matrix2.cols(); j++)
			{
				double result = matrix1.row(i).dot(matrix2.col(j));
				output(i, j) = result;
			}
		}

		return output;
	}

	double sigmoid(double x)
	{
		return 1.0 / (1.0 + exp(-x));;
	}

	double sigmoid_derivative(double x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
	}

	double result_difference(double actual, double predicted)
	{
		return predicted - actual;
	}
}