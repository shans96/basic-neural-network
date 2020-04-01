#include "../lib/Eigen/Dense"
#include <vector>
#include <random>

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
        std::vector<int> layers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biases;
        double sigmoid_activation(double x);
};