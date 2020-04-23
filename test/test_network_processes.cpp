#include "catch.hpp"
#include "../src/Network.hpp"

const double ACCURACY = 0.000001;
const Eigen::MatrixXd SMALL_NETWORK_INPUT = Eigen::Vector2d(2.0, 1.0);
const double SMALL_NETWORK_EXPECTED_OUTPUT = 1.0;

std::vector<Eigen::MatrixXd> create_small_weights_set(double first_layer[], double second_layer[])
{
	Eigen::MatrixXd first_layer_weights(3, 2);
	first_layer_weights << first_layer[0], first_layer[1],
		first_layer[2], first_layer[3],
		first_layer[4], first_layer[5];

	Eigen::MatrixXd second_layer_weights(1, 3);
	second_layer_weights << second_layer[0], second_layer[1], second_layer[2];

	std::vector<Eigen::MatrixXd> weights;
	weights.reserve(2);
	weights.push_back(first_layer_weights);
	weights.push_back(second_layer_weights);

	return weights;
}

std::vector<Eigen::MatrixXd> create_small_biases_set(double first_layer[], double second_layer[])
{
	Eigen::MatrixXd first_layer_biases(3, 1);
	first_layer_biases << first_layer[0],
		first_layer[1],
		first_layer[2];

	Eigen::MatrixXd second_layer_biases(1, 1);
	second_layer_biases << second_layer[0];

	std::vector<Eigen::MatrixXd> biases;
	biases.reserve(2);
	biases.push_back(first_layer_biases);
	biases.push_back(second_layer_biases);

	return biases;
}

std::vector<Eigen::MatrixXd> create_weights()
{
	double first_layer[6] = { -0.19964734, -0.22955651,
		-0.69292439, -2.67396935,
		0.09088148, 0.58786673 
	};
	double second_layer[3] = { 0.5958179, 0.12265036, 0.67701942 };

	return create_small_weights_set(first_layer, second_layer);
}

std::vector<Eigen::MatrixXd> create_biases()
{
	double first_layer[3] = { 0.21737026, 0.20908332, 1.66283142 };
	double second_layer[1] = { -1.85470767 };

	return create_small_biases_set(first_layer, second_layer);
}

std::vector<Eigen::MatrixXd> create_small_backprop_weights()
{
	double first_layer[6] = { -0.04111753, -0.02055877,
		-0.00071989, -0.00035995,
		-0.01446475, -0.00723237 
	};
	double second_layer[3] = { -0.05737056,
		-0.00299714,
		-0.13232482 
	};

	return create_small_weights_set(first_layer, second_layer);
}

std::vector<Eigen::MatrixXd> create_small_backprop_biases()
{
	double first_layer [3]= { -0.02055877,
		-0.00035995,
		-0.00723237 
	};
	double second_layer[1] = { -0.14394565 };

	return create_small_biases_set(first_layer, second_layer);
}

std::vector<Eigen::MatrixXd> create_small_trained_weights()
{
	double first_layer[6] = { -0.1984116, -0.22893864,
		-0.69290277, -2.67395854,
		0.09131636, 0.58808417 
	};
	double second_layer[3] = { 0.59754129,
		0.12274033,
		0.6809916 
	};

	return create_small_weights_set(first_layer, second_layer);
}

std::vector<Eigen::MatrixXd> create_small_trained_biases()
{
	double first_layer[3] = { 0.21798813,
		0.20909413,
		1.66304886 
	};
	double second_layer[1] = { -1.85038681 };

	return create_small_biases_set(first_layer, second_layer);
}

Network create_test_network()
{
	std::vector<int> layers = { 2,3,1 };
	std::vector<Eigen::MatrixXd> weights = create_weights();
	std::vector<Eigen::MatrixXd> biases = create_biases();

	return Network(layers, weights, biases);
}

std::vector<xy_data> create_small_training_dataset()
{
	std::vector<xy_data> training_data;
	Eigen::MatrixXd expected_result(1, 1);
	expected_result << SMALL_NETWORK_EXPECTED_OUTPUT;
	training_data.push_back(std::make_pair(SMALL_NETWORK_INPUT, expected_result));
	return training_data;
}

TEST_CASE("Feed forward produces the correct (approximate) answer", "[feedforward]")
{
	Network network = create_test_network();
	double network_feed_forward_result = std::get<0>(network.feed_forward(Eigen::Vector2d(2.0, 1.0)))(0);
	WARN("Network feed forward output: " << network_feed_forward_result << ". Expecting approximately 0.27045343.");
	REQUIRE(network_feed_forward_result == Approx(0.27045343).epsilon(ACCURACY));
}

TEST_CASE("Backpropagation produces the correct (approximate) weight and bias values across all neurons after a single pass", "[backpropagation]")
{
	Network network = create_test_network();
	Eigen::MatrixXd expected_output(1, 1);
	expected_output << 1;

	std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> backpropagation_result = network.backpropagate(SMALL_NETWORK_INPUT, expected_output);

	std::vector<Eigen::MatrixXd> expected_weights = create_small_backprop_weights();
	std::vector<Eigen::MatrixXd> expected_biases = create_small_backprop_biases();

	std::vector<Eigen::MatrixXd> calculated_biases = std::get<0>(backpropagation_result);
	std::vector<Eigen::MatrixXd> calculated_weights = std::get<1>(backpropagation_result);

	REQUIRE(calculated_biases[0].isApprox(expected_biases[0], ACCURACY));
	REQUIRE(calculated_biases[1].isApprox(expected_biases[1], ACCURACY));
	REQUIRE(calculated_weights[0].isApprox(expected_weights[0], ACCURACY));
	REQUIRE(calculated_weights[1].isApprox(expected_weights[1], ACCURACY));
}

TEST_CASE("Gradient descent produces specific weight and bias values after a certain amount of epochs", "[gradient_descent]")
{
	Network network = create_test_network();
	network.mini_batch_gradient_descent(0.01, 3, 3, create_small_training_dataset());
	std::vector<Eigen::MatrixXd> network_weights = network.get_weights();
	std::vector<Eigen::MatrixXd> network_biases = network.get_biases();

	std::vector<Eigen::MatrixXd> expected_trained_biases = create_small_trained_biases();
	std::vector<Eigen::MatrixXd> expected_trained_weights = create_small_trained_weights();

	REQUIRE(expected_trained_weights[0].isApprox(network_weights[0], ACCURACY));
	REQUIRE(expected_trained_weights[1].isApprox(network_weights[1], ACCURACY));
	REQUIRE(expected_trained_biases[0].isApprox(network_biases[0], ACCURACY));
	REQUIRE(expected_trained_biases[1].isApprox(network_biases[1], ACCURACY));
}