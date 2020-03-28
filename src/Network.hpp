#include <vector>
#include <random>

class Network 
{
    public:
        Network(const std::vector<int> layer_data);

    private:
        // Use uniform distribution (for now)
        std::default_random_engine generator;
        std::normal_distribution<float> distribution;
        int layers;
        void generate_weights();
        void generate_biases();
        std::vector<int> sizes;
        // Structure is weights[layer][neuron][weight]
        std::vector<std::vector<std::vector<float>>> weights;
        // Structure is biases[layer][bias]
        std::vector<std::vector<float>> biases;
        float sigmoid_activation(float x);
};