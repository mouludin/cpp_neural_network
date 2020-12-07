#pragma once

// activation function
#define ACT_SIGMOID 1
#define ACT_TANH 2
#define ACT_RELU 3


#include "matrix.hpp"

class layers
{
public:
    unsigned short size;
    std::vector<unsigned short> layers_data;
    std::vector<unsigned short> activations;

private:
    unsigned short remaining_empty_layers;

public:
    layers(unsigned short size);

    void in(unsigned short num_of_layer, unsigned short activation = ACT_SIGMOID);
};

class NeuralNetwork
{
private:
    std::vector<unsigned short> num_layers;
    std::vector<unsigned short> activation;
    Matrix** weight_data;
    Matrix** bias_data;
    unsigned short layer_size;
    float learning_rate;
    unsigned short weight_data_length = 0;

public:
    NeuralNetwork(layers m_layer);

    std::vector<float> predict(std::vector<float> data);

    void mutate(float rate);

    void train(std::vector<std::vector<float>> xs, std::vector<std::vector<float>> ys, unsigned int ephocs);

    void export_neuron(const char* filename = "neuron_data.bin");
    void import_neuron(const char* filename);
    ~NeuralNetwork();

private:
    float getActivation(unsigned short activation, float value);
    float getDerivative(unsigned short activation, float value);
};