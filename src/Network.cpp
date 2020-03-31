#include "Network.hpp"

Network::Network(const std::vector<int> layer_data)
    : weights(0), biases(0)
{
    layers = layer_data.size();
    sizes = layer_data;
    generate_weights();
	generate_biases();
}

void Network::generate_biases()
{
	// Input layer has no biases, splice it
	std::vector<int> temp = sizes;
	temp.erase(temp.begin());
	for (size_t i = 0; i < temp.size(); i++)
	{
		biases.push_back(Eigen::MatrixXd::Random(temp[i], 1));
	}
}

void Network::generate_weights()
{
	std::vector<int>::iterator it;
	// Input layer has no weights, subtract 1
	for (auto it = sizes.begin(); it < sizes.end() - 1; it++)
	{
		weights.push_back(Eigen::MatrixXd::Random(*it, *next(it)));
	}
}

float Network::sigmoid_activation(float x)
{
    return 1.0/(1.0 + exp(-x));
}