#include "catch.hpp"
#include "../src/Network.hpp"

const double ACCURACY = 0.000001;
const Eigen::MatrixXd SMALL_NETWORK_INPUT = Eigen::Vector2d(2.0, 1.0);
const double SMALL_NETWORK_EXPECTED_OUTPUT = 1.0;

std::vector<Eigen::MatrixXd> create_weights()
{
	Eigen::MatrixXd first_layer_weights(3, 2);
	first_layer_weights << -0.19964734, -0.22955651,
		-0.69292439, -2.67396935,
		0.09088148, 0.58786673;

	Eigen::MatrixXd second_layer_weights(1, 3);
	second_layer_weights << 0.5958179, 0.12265036, 0.67701942;

	std::vector<Eigen::MatrixXd> weights;
	weights.reserve(2);
	weights.push_back(first_layer_weights);
	weights.push_back(second_layer_weights);

	return weights;
}

std::vector<Eigen::MatrixXd> create_biases()
{
	Eigen::MatrixXd first_layer_biases(3, 1);
	first_layer_biases << 0.21737026,
		0.20908332,
		1.66283142;

	Eigen::MatrixXd second_layer_biases(1, 1);
	second_layer_biases << -1.85470767;

	std::vector<Eigen::MatrixXd> biases;
	biases.reserve(2);
	biases.push_back(first_layer_biases);
	biases.push_back(second_layer_biases);

	return biases;
}

std::vector<Eigen::MatrixXd> create_small_backprop_weights()
{
	std::vector<Eigen::MatrixXd> expected_weights;

	Eigen::MatrixXd first_weights(3,2);
	Eigen::MatrixXd second_weights(1,3);

	first_weights << -0.04111753, -0.02055877,
		-0.00071989, -0.00035995,
		-0.01446475, -0.00723237;

	second_weights << -0.05737056, 
		-0.00299714, 
		-0.13232482;

	expected_weights.push_back(first_weights);
	expected_weights.push_back(second_weights);

	return expected_weights;
}

std::vector<Eigen::MatrixXd> create_small_backprop_biases()
{
	std::vector<Eigen::MatrixXd> expected_biases;
	Eigen::MatrixXd first_biases(3,1);
	Eigen::MatrixXd second_biases(1,1);

	first_biases << -0.02055877,
		-0.00035995,
		-0.00723237;

	second_biases << -0.14394565;

	expected_biases.push_back(first_biases);
	expected_biases.push_back(second_biases);

	return expected_biases;
}

std::vector<Eigen::MatrixXd> create_small_trained_weights()
{
	std::vector<Eigen::MatrixXd> trained_weights;

	Eigen::MatrixXd first_weights(3, 2);
	Eigen::MatrixXd second_weights(1, 3);

	first_weights << -0.1984116, -0.22893864,
		-0.69290277, -2.67395854,
		0.09131636, 0.58808417;

	second_weights << 0.59754129,
		0.12274033,
		0.6809916;

	trained_weights.push_back(first_weights);
	trained_weights.push_back(second_weights);

	return trained_weights;
}

std::vector<Eigen::MatrixXd> create_small_trained_biases()
{
	std::vector<Eigen::MatrixXd> trained_biases;
	Eigen::MatrixXd first_biases(3, 1);
	Eigen::MatrixXd second_biases(1, 1);

	first_biases << 0.21798813,
		0.20909413,
		1.66304886;

	second_biases << -1.85038681;

	trained_biases.push_back(first_biases);
	trained_biases.push_back(second_biases);

	return trained_biases;
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