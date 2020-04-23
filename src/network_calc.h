#include "../lib/Eigen/Dense"
#include <vector>


namespace network_calc 
{
	Eigen::MatrixXd multiply_matrices(Eigen::MatrixXd matrix1, Eigen::MatrixXd matrix2);
	std::vector<Eigen::MatrixXd> created_zeroed_layers(std::vector<Eigen::MatrixXd> *layers_to_clone);
	double sigmoid(double x);
	double sigmoid_derivative(double x);
	double result_difference(double actual, double predicted);
}