#include "Network.hpp"

Network::Network(const std::vector<int> layer_data)
    : weights(0), biases(0)
{
    layers = layer_data;
    generate_weights();
	generate_biases();
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

float Network::sigmoid_activation(float x)
{
    return 1.0/(1.0 + exp(-x));
}