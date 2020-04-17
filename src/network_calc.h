#include "../lib/Eigen/Dense"
#include <vector>


namespace network_calc 
{
	Eigen::MatrixXd multiply_matrices(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2);
	double calculate_error(Eigen::MatrixXd expected, Eigen::MatrixXd calculated);
	double sigmoid(double x);
	double sigmoid_derivative(double x);
	double result_difference(double actual, double predicted);
}