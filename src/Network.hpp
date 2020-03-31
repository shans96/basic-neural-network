#include "../lib/Eigen/Dense"
#include <vector>
#include <random>

class Network 
{
    public:
        Network(const std::vector<int> layer_data);

    private:
        void generate_weights();
        void generate_biases();
        std::vector<int> layers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biases;
        double sigmoid_activation(double x);
};