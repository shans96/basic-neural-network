#include "../lib/Eigen/Dense"
#include "network_calc.h"
#include <functional>
#include <valarray>
#include <vector>

class Network 
{
    public:
        Network(const std::vector<int> layer_data);
        Network(const std::vector<int> layer_data, 
            std::vector<Eigen::MatrixXd> weights, std::vector<Eigen::MatrixXd> biases);
        Eigen::MatrixXd feed_forward(Eigen::VectorXd input);

    private:
        void generate_weights();
        void generate_biases();
        void initialize_layers(std::vector<int> layers);
        std::vector<int> layers;
        std::vector<int> non_input_layers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biases;
};