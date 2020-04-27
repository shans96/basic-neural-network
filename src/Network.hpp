#include "../lib/Eigen/Dense"
#include "network_calc.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <utility>
#include <valarray>
#include <vector>

typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> xy_data;

class Network 
{
    public:
        Network(const std::vector<int> layer_data);
        Network(const std::vector<int> layer_data, 
            std::vector<Eigen::MatrixXd> weights, std::vector<Eigen::MatrixXd> biases);
        std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> feed_forward(Eigen::VectorXd input);
        std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> backpropagate(Eigen::MatrixXd input, Eigen::MatrixXd expected_output);
        void mini_batch_gradient_descent(double alpha, int epochs, int batch_size, std::vector<xy_data> training_data);
        std::vector<Eigen::MatrixXd> get_weights();
        std::vector<Eigen::MatrixXd> get_biases();
        std::vector<int> get_layers();

    private:
        void generate_weights();
        void generate_biases();
        void initialize_layers(std::vector<int> layers);
        std::vector<int> layers;
        std::vector<int> non_input_layers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biases;
        void update_weights_biases(std::vector<xy_data>, double alpha);
};