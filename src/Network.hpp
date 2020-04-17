#include "../lib/Eigen/Dense"
#include "network_calc.h"
#include <functional>
#include <tuple>
#include <iostream>
#include <utility>
#include <valarray>
#include <vector>

typedef std::pair<std::vector<double>, std::vector<double>> xy_data;

class Network 
{
    public:
        Network(const std::vector<int> layer_data);
        Network(const std::vector<int> layer_data, 
            std::vector<Eigen::MatrixXd> weights, std::vector<Eigen::MatrixXd> biases);
        std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> feed_forward(Eigen::VectorXd input);
        std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> backpropagate(Eigen::MatrixXd input, Eigen::MatrixXd expected_output);

    private:
        void generate_weights();
        void generate_biases();
        void initialize_layers(std::vector<int> layers);
        std::vector<int> layers;
        std::vector<int> non_input_layers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biases;
        void mini_batch_gradient_descent(double alpha, int epochs, int batch_size, std::vector<xy_data> training_data);
};