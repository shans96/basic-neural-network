#include "Network.hpp"

Network::Network(const std::vector<int> layer_data)
    : weights(0), biases(0)
{
	initialize_layers(layer_data);
    generate_weights();
	generate_biases();
}

Network::Network(const std::vector<int> layer_data, 
	std::vector<Eigen::MatrixXd> network_weights, std::vector<Eigen::MatrixXd> network_biases)
{
	initialize_layers(layer_data);
	weights = network_weights;
	biases = network_biases;
}

void Network::initialize_layers(std::vector<int> layer_data)
{
	layers = layer_data;
	non_input_layers = std::vector<int>(layers.begin() + 1, layers.end());
}

void Network::generate_biases()
{
	biases.reserve(non_input_layers.size());
	for (size_t i = 0; i < non_input_layers.size(); i++)
	{
		biases.push_back(Eigen::MatrixXd::Random(non_input_layers[i], 1));
	}
}

void Network::generate_weights()
{
	weights.reserve(layers.size());
	for (auto it = layers.begin(); it != layers.end(); it++)
	{
		weights.push_back(Eigen::MatrixXd::Random(*it, *next(it)));
	}
}

Eigen::MatrixXd Network::feed_forward(Eigen::VectorXd input)
{
	for (size_t i = 0; i < non_input_layers.size(); i++)
	{
		input = network_calc::multiply_matrices(weights[i], input) + biases[i];
		input = input.unaryExpr([] (double x) {
				return 1.0 / (1.0 + exp(-x));
			});
	}
	return input;
}
