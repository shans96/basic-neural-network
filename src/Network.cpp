#include "Network.hpp"

Network::Network(const std::vector<int> layer_data)
    : weights(0), biases(0)
{
    layers = layer_data;
    generate_weights();
	generate_biases();
}

Network::Network(const std::vector<int> layer_data, 
	std::vector<Eigen::MatrixXd> network_weights, std::vector<Eigen::MatrixXd> network_biases)
{
	layers = layer_data;
	weights = network_weights;
	biases = network_biases;
}

void Network::generate_biases()
{
	// Input layer has no biases, splice it
	std::vector<int> temp = layers;
	temp.erase(temp.begin());
	biases.reserve(temp.size());
	for (size_t i = 0; i < temp.size(); i++)
	{
		biases.push_back(Eigen::MatrixXd::Random(temp[i], 1));
	}
}

void Network::generate_weights()
{
	std::vector<int>::iterator it;
	weights.reserve(layers.size());
	// Input layer has no weights, subtract 1
	for (auto it = layers.begin(); it < layers.end() - 1; it++)
	{
		weights.push_back(Eigen::MatrixXd::Random(*it, *next(it)));
	}
}

Eigen::MatrixXd Network::feed_forward(Eigen::VectorXd input)
{
	// Returns the output of the entire network.
	// Note: Eigen doesn't support multidimensional matrix products,
	// so activations have to be calculated individually and then be placed
	// into a new matrix.
	std::vector<int> temp(layers);
	temp.erase(temp.begin());
	// For each layer
	for (size_t i = 0; i < temp.size(); i++)
	{
		Eigen::VectorXd layer_output(temp[i]);
		// For each neuron in the layer
		for (size_t j = 0; j < weights[i].rows(); j++)
		{
			Eigen::RowVectorXd neuron_weights = weights[i].row(j);
			double neuron_output = sigmoid_activation(neuron_weights.dot(input) + biases[i](j));
			layer_output(j, 0) = neuron_output;
		}
		input = layer_output;
	}
	
	return input;
}

double Network::sigmoid_activation(double x)
{
    return 1.0/(1.0 + exp(-x));
}