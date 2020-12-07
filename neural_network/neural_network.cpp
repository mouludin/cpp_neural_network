#include <cmath>
#include <ctime>
#include <fstream>
#include "neural_network.hpp"

layers::layers(unsigned short size)
    :size(size), remaining_empty_layers(size) {}

void layers::in(unsigned short num_of_layer, unsigned short activation) 
{
    if (remaining_empty_layers > 0)
    {
        layers_data.push_back(num_of_layer);
        activations.push_back(activation);
        remaining_empty_layers -= 1;
    }
}


NeuralNetwork::NeuralNetwork(layers m_layer) 
    :num_layers(m_layer.layers_data), activation(m_layer.activations), layer_size(m_layer.size)
{
    srand((unsigned int)time(NULL));

    weight_data = new Matrix*;
    bias_data = new Matrix*;

    learning_rate = 0.1f;

    for (short layer = 0; layer < layer_size - 1; layer++)
    {
        weight_data[layer] = new Matrix(num_layers[layer + 1], num_layers[layer]);
        weight_data[layer]->SetRandom();
        bias_data[layer] = new Matrix(num_layers[layer + 1], 1);
        bias_data[layer]->SetRandom();
        weight_data_length++;
    }
}

std::vector<float> NeuralNetwork::predict(std::vector<float> data)
{
    std::vector<std::vector<float>> layer_data;
    // layer data di tambah input
    layer_data.push_back(data);
    for (unsigned short i = 1; i < num_layers.size(); i++)
    {
        // membuat layer selanjutnya
        std::vector<float> nextlayer;

        // mengisi hasil layer selanjutnya
        for (unsigned short j = 0; j < num_layers[i]; j++)
        {
            float value = 0;
            for (unsigned short k = 0; k < num_layers[i - 1]; k++)
            {
                value += layer_data[i - 1][k] * weight_data[i - 1]->arr2d[j][k];
            }
            value += bias_data[i - 1]->arr2d[j][0];
            value = getActivation(activation[i], value);
            nextlayer.push_back(value);
        }

        // menambahkan layer selanjutnya
        layer_data.push_back(nextlayer);
    }

    // menghasilkan layer terakhir
    return layer_data.back();
}

void NeuralNetwork::mutate(float rate)
{
    for (unsigned short i = 1; i < num_layers.size(); i++)
    {
        for (unsigned short j = 0; j < num_layers[i]; j++)
        {
            for (unsigned short k = 0; k < num_layers[i - 1]; k++)
            {
                float random_value = (rand() % 100) * 0.01f;
                if ((rand() % 100) * 0.01f < rate)
                {
                    float random_value = (rand() % 100) * 0.01f;
                    weight_data[i - 1]->arr2d[j][k] = random_value;
                }
            }
        }
    }
}

void NeuralNetwork::train(std::vector<std::vector<float>> xs, std::vector<std::vector<float>> ys, unsigned int ephocs) 
{
    for (unsigned int eph = 0; eph < ephocs; eph++)
    {
        for (unsigned short x = 0; x < xs.size(); x++)
        {
            // layer data di tambah input
            std::vector<std::vector<std::vector<float>>> layers;
            layers.push_back(Matrix::array2d(xs[x]));
            for (unsigned short i = 1; i < num_layers.size(); i++)
            {
                // membuat layer selanjutnya
                std::vector<std::vector<float>> nextlayer;

                // mengisi hasil layer selanjutnya
                for (unsigned short j = 0; j < num_layers[i]; j++)
                {
                    float value = 0;
                    for (unsigned short k = 0; k < num_layers[i - 1]; k++)
                    {
                        value += layers[i - 1][k][0] * weight_data[i - 1]->arr2d[j][k];
                    }
                    value += bias_data[i - 1]->arr2d[j][0];
                    value = getActivation(activation[i], value);
                    std::vector<float> result;
                    result.push_back(value);
                    nextlayer.push_back(result);
                }

                // menambahkan layer selanjutnya
                layers.push_back(nextlayer);
            }

            std::vector<std::vector<float>> predict_value = layers.back();
            std::vector<std::vector<float>> loss;
            for (unsigned short i = 0; i < predict_value.size(); i++)
            {
                std::vector<float> l;
                l.push_back(ys[x][i] - predict_value[i][0]);
                loss.push_back(l);
            }

            // backpropagation
            for (short b = weight_data_length - 1; b >= 0; --b)
            {
                std::vector<std::vector<float>> optimizer;
                for (std::vector<float> predict_val : predict_value)
                {
                    std::vector<float> setOnDerivative;
                    setOnDerivative.push_back(getDerivative(activation[b], predict_val[0]));
                    optimizer.push_back(setOnDerivative);
                }

                for (unsigned short i = 0; i < optimizer.size(); i++)
                {
                    optimizer[i][0] *= loss[i][0] * learning_rate;
                }

                std::vector<std::vector<float>> layer_T = Matrix::transpose(layers[b]);
                std::vector<std::vector<float>> w_layers_deltas = Matrix::multiplyArray2d(optimizer, layer_T);

                weight_data[b]->add(w_layers_deltas);
                bias_data[b]->add(optimizer);

                std::vector<std::vector<float>> weight_T = Matrix::transpose(weight_data[b]->arr2d);
                std::vector<std::vector<float>> layer_error = Matrix::multiplyArray2d(weight_T, loss);

                predict_value = layers[b];
                loss = layer_error;
            }
        }
    }
}

NeuralNetwork::~NeuralNetwork()
{}

float NeuralNetwork::getActivation(unsigned short activation, float value)
{
    float result;
    if (activation == ACT_SIGMOID)
    {
        result = 1 / (1 + exp(value * -1));
    }
    else if (activation == ACT_TANH)
    {
        result = (exp(value) - exp(value * -1)) / (exp(value) + exp(value * -1));
    }
    else if (activation == ACT_RELU)
    {
        if(value <= 0){
            result = 0.0f;
        }else{
            result = value;
        }
    }
    
    return result;
}

float NeuralNetwork::getDerivative(unsigned short activation, float value)
{
    float result;
    if (activation == ACT_SIGMOID)
    {
        result = value * (1 - value);
    }
    else if (activation == ACT_TANH)
    {
        result = 1 - (value * value);
    }
    else if (activation == ACT_RELU)
    {
        if(value > 0){
            result = 1.0f;
        }else{
            result = 0.0f;
        }
    }
    return result;
}

void NeuralNetwork::export_neuron(const char* filename)
{
    std::fstream myFile;

    myFile.open(filename, std::ios::out | std::ios::binary);
    for (short layer = 0; layer < layer_size - 1; layer++)
    {
        for (short i = 0; i < weight_data[layer]->arr2d.size(); i++)
            for (short j = 0; j < weight_data[layer]->arr2d[i].size(); j++)
                myFile.write((char*)&weight_data[layer]->arr2d[i][j], sizeof(weight_data[layer]->arr2d[i][j]));

        for (short i = 0; i < bias_data[layer]->arr2d.size(); i++)
            for (short j = 0; j < bias_data[layer]->arr2d[i].size(); j++)
                myFile.write((char*)&bias_data[layer]->arr2d[i][j], sizeof(bias_data[layer]->arr2d[i][j]));
    }
    myFile.close();
}

void NeuralNetwork::import_neuron(const char* filename)
{
    std::fstream myFile;
    myFile.open(filename, std::ios::in | std::ios::binary);
    for (short layer = 0; layer < layer_size - 1; layer++)
    {
        for (short i = 0; i < weight_data[layer]->arr2d.size(); i++)
            for (short j = 0; j < weight_data[layer]->arr2d[i].size(); j++)
                myFile.read((char*)&weight_data[layer]->arr2d[i][j], sizeof(weight_data[layer]->arr2d[i][j]));

        for (short i = 0; i < bias_data[layer]->arr2d.size(); i++)
            for (short j = 0; j < bias_data[layer]->arr2d[i].size(); j++)
                myFile.read((char*)&bias_data[layer]->arr2d[i][j], sizeof(bias_data[layer]->arr2d[i][j]));
    }
    myFile.close();
}