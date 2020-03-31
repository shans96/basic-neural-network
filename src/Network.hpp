#include "../lib/Eigen/Dense"
#include <vector>
#include <random>

class Network 
{
    public:
        Network(const std::vector<int> layer_data);

    private:
        int layers;
        void generate_weights();
        void generate_biases();
        std::vector<int> sizes;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::MatrixXd> biases;
        float sigmoid_activation(float x);
};