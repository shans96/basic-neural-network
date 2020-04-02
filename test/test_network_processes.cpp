#include "catch.hpp"
#include "../src/Network.hpp"

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

Network create_test_network()
{
	std::vector<int> layers = { 2,3,1 };
	std::vector<Eigen::MatrixXd> weights = create_weights();
	std::vector<Eigen::MatrixXd> biases = create_biases();

	return Network(layers, weights, biases);
}


TEST_CASE("Neural network processes produce the correct (approximate) answers", "[processes]")
{
	Network network = create_test_network();
	double network_feed_forward_result = network.feed_forward(Eigen::Vector2d(2.0, 1.0))(0);
	WARN("Network feed forward output: " << network_feed_forward_result << ". Expecting approximately 0.27045343.");
	REQUIRE(network_feed_forward_result == Approx(0.27045343).epsilon(0.000001));
}
