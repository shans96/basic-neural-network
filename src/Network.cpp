#include "Network.hpp"

Network::Network(const std::vector<int> layer_data)
    : biases(layer_data.size() - 1), distribution(0, 1)
{
    layers = layer_data.size();
    sizes = layer_data;
    generate_biases();
    generate_weights();
}

void Network::generate_biases()
{
    // First layer has no biases so we skip it
    std::vector<int> temp = sizes;
    temp.erase(temp.begin());
    for (size_t i = 0; i < temp.size(); i++)
    {
        for (size_t j = 0; j < temp[i]; j++)
        {
            biases[i].push_back(distribution(generator));
        }
    }
}

void Network::generate_weights()
{
    // By copying the array and slicing one at the beginning, one at the end,
    // you get the matrix sizes of the weights
    std::vector<int> temp2 = sizes;
    std::vector<int> temp3 = sizes;
    temp2.erase(temp2.begin());
    temp3.pop_back();

    std::vector<std::vector<float>> layer_weights;
    std::vector<float> neuron_weights;

    size_t i = 0;
    for (size_t j = 0; j < sizes.size() - 1; j++)
    {
        layer_weights.clear();
        for (size_t k = 0; k < temp3[i]; k++)
        {
            neuron_weights.clear();
            for (size_t l = 0; l < temp2[i]; l++)
            {
                neuron_weights.push_back(distribution(generator));
            }
            layer_weights.push_back(neuron_weights);
        }
        weights.push_back(layer_weights);
        i++;
    }
}

float Network::sigmoid_activation(float x)
{
    return 1.0/(1.0 + exp(-x));
}