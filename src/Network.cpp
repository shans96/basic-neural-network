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
	weights.reserve(layers.size() - 1);
	for (auto it = layers.begin(); it != layers.end() - 1; it++)
	{
		weights.push_back(Eigen::MatrixXd::Random(*it, *next(it)));
	}
}

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::feed_forward(Eigen::VectorXd input)
{
	std::vector<Eigen::MatrixXd> net_inputs;
	std::vector<Eigen::MatrixXd> activations;
	net_inputs.reserve(non_input_layers.size() - 1);
	activations.reserve(non_input_layers.size());
	activations.push_back(input);
	for (size_t i = 0; i < non_input_layers.size(); i++)
	{
		input = network_calc::multiply_matrices(weights[i], input) + biases[i];
		net_inputs.push_back(input);
		input = input.unaryExpr([] (double x) {
				return 1.0 / (1.0 + exp(-x));
			});
		activations.push_back(input);
	}
	return std::make_tuple(input, net_inputs, activations);
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::backpropagate(Eigen::MatrixXd input, Eigen::MatrixXd expected_output)
{
	auto feed_forward_results = (*this).feed_forward(input);
	Eigen::MatrixXd calculated_output = std::get<0>(feed_forward_results);
	std::vector<Eigen::MatrixXd> net_inputs = std::get<1>(feed_forward_results);
	std::vector<Eigen::MatrixXd> activations = std::get<2>(feed_forward_results);

	std::vector<Eigen::MatrixXd> cloned_weights(weights);
	std::vector<Eigen::MatrixXd> cloned_biases(biases);

	for (size_t i = 0; i < layers[layers.size() - 1]; i++)
	{
		cloned_biases[cloned_biases.size() - 1](i, 0) = network_calc::result_difference(expected_output(i, 0), calculated_output(i, 0)) * 
			network_calc::sigmoid_derivative(net_inputs[net_inputs.size() - 1](i));
	}

	Eigen::MatrixXd delta = cloned_biases[cloned_biases.size() - 1];

	cloned_weights[cloned_weights.size() - 1] = network_calc::multiply_matrices(delta, activations[activations.size() - 2].transpose());

	for (size_t i = non_input_layers.size() - 1; !(i == 0); i--)
	{
		Eigen::MatrixXd partial_calc = net_inputs[i - 1].unaryExpr([] (double x) {
			return network_calc::sigmoid_derivative(x);
		});

		delta  = network_calc::multiply_matrices(weights[i].transpose(), delta).cwiseProduct(partial_calc);

		cloned_biases[i - 1] = delta; 
		cloned_weights[i - 1] = network_calc::multiply_matrices(delta, activations[i - 1].transpose());
	}

	return std::make_pair(cloned_biases, cloned_weights);
}

void Network::mini_batch_gradient_descent(double alpha, int epochs, int batch_size, std::vector<xy_data> training_data)
{
	auto rng = std::default_random_engine{};
	std::vector<xy_data> mini_batch;
	mini_batch.reserve(batch_size);
	if (training_data.size() < batch_size)
	{
		std::cout << "Warning: training data size is less than the batch size. Batch size will be changed to match training data size.\n";
		batch_size = training_data.size();
	}
	for (size_t i = 0; i < epochs; i++)
	{
		std::shuffle(std::begin(training_data), std::end(training_data), rng);
		for (size_t j = 0; j < batch_size; j++)
		{
			mini_batch.push_back(training_data[j]);;
		}

		update_weights_biases(mini_batch, alpha);
		// Perform prediction on the last example just to check
		Eigen::MatrixXd new_predicted_output = std::get<2>((*this).feed_forward(training_data.back().first)).back();

		std::cout << "Epoch " << std::to_string(i) << ", "
			<< "SSE: " << std::to_string(network_calc::sum_squared_error(new_predicted_output, training_data.back().second)) << ".\n";
	}
}

std::vector<Eigen::MatrixXd> Network::get_weights()
{
	return weights;
}

std::vector<Eigen::MatrixXd> Network::get_biases()
{
	return biases;
}

std::vector<int> Network::get_layers()
{
	return layers;
}

void Network::update_weights_biases(std::vector<xy_data> batch, double alpha)
{
	std::vector<Eigen::MatrixXd> delta_weights = network_calc::created_zeroed_layers(&weights);
	std::vector<Eigen::MatrixXd> delta_biases = network_calc::created_zeroed_layers(&biases);
	for (size_t i = 0; i < batch.size(); i++)
	{
		auto backprop_output_pair = (*this).backpropagate(batch[i].first, batch[i].second);
		for (size_t j = 0; j < backprop_output_pair.first.size(); j++)
		{
			delta_weights[j] += backprop_output_pair.second[j];
			delta_biases[j] += backprop_output_pair.first[j];
		}
	}

	for (size_t i = 0; i < weights.size(); i++)
	{
		weights[i] -= (alpha / batch.size()) * delta_weights[i];
		biases[i] -= (alpha / batch.size()) * delta_biases[i];
	}
}
