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

Datasets::Datasets(const char* filename)
{
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>


    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int val;

    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        // Extract each column name
        while(std::getline(ss, colname, ',')){
            // Initialize and add <colname, int vector> pairs to m_data
            m_data.push_back({colname, std::vector<float> {}});
        }
    }

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val){
            // Add the current integer to the 'colIdx' column's values vector
            m_data[colIdx].second.push_back((float)val);

            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }
    }

    // Close file
    myFile.close();
}


NeuralNetwork::NeuralNetwork(layers& m_layer) 
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

std::vector<float> NeuralNetwork::predict(std::vector<float>& data)
{
    MATRIX(float) layer_data;
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

void NeuralNetwork::train(MATRIX(float)& xs, MATRIX(float)& ys, unsigned int ephocs) 
{
    for (unsigned int eph = 0; eph < ephocs; eph++)
    {
        for (unsigned short x = 0; x < xs.size(); x++)
        {
            // layer data di tambah input
            std::vector<MATRIX(float)> layers;
            layers.push_back(Matrix::tomatrix(xs[x]));
            for (unsigned short i = 1; i < num_layers.size(); i++)
            {
                // membuat layer selanjutnya
                MATRIX(float) nextlayer;

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

            MATRIX(float) predict_value = layers.back();
            MATRIX(float) loss;
            for (unsigned short i = 0; i < predict_value.size(); i++)
            {
                std::vector<float> l;
                l.push_back(ys[x][i] - predict_value[i][0]);
                loss.push_back(l);
            }

            // backpropagation
            for (short b = weight_data_length - 1; b >= 0; --b)
            {
                MATRIX(float) optimizer;
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

                MATRIX(float) layer_T = Matrix::transpose(layers[b]);
                MATRIX(float) w_layers_deltas = Matrix::multiplyMatrix(optimizer, layer_T);

                weight_data[b]->add(w_layers_deltas);
                bias_data[b]->add(optimizer);

                MATRIX(float) weight_T = Matrix::transpose(weight_data[b]->arr2d);
                MATRIX(float) layer_error = Matrix::multiplyMatrix(weight_T, loss);

                predict_value = layers[b];
                loss = layer_error;
            }
        }
    }
}

void NeuralNetwork::train(Datasets datasets,std::vector<std::string> colname_input,std::vector<std::string> colname_output, unsigned int ephocs)
{
    MATRIX(float) xs;
    MATRIX(float) ys;
    // train(xs,ys,ephocs);
    for(short i = 0;i < colname_input.size();i++)
    {
        for(short j = 0;j < datasets.m_data.size();j++){
            if(datasets.m_data[j].first == colname_input[i]){
                for(short val = 0;val < datasets.m_data.at(j).second.size();val++)
                {
                    if(i == 0) xs.push_back(std::vector<float> {datasets.m_data.at(j).second[val]});
                    else xs[val].push_back(datasets.m_data.at(j).second[val]);
                }
            }
        }
    }

    for(short i = 0;i < colname_output.size();i++)
    {
        for(short j = 0;j < datasets.m_data.size();j++){
            if(datasets.m_data[j].first == colname_output[i]){
                for(short val = 0;val < datasets.m_data.at(j).second.size();val++)
                {
                    if(i == 0) ys.push_back(std::vector<float> {datasets.m_data[j].second[val]});
                    else ys[val].push_back(datasets.m_data.at(j).second[val]);
                }
            }
        }
    }

    train(xs,ys,ephocs);
}

NeuralNetwork::~NeuralNetwork()
{
    for (short layer = 0; layer < layer_size - 1; layer++)
    {
        delete[] weight_data[layer];
        delete[] bias_data[layer];
    }
    delete[] weight_data;
    delete[] bias_data;
}

float NeuralNetwork::getActivation(unsigned short& activation, float& value) const
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

float NeuralNetwork::getDerivative(unsigned short& activation, float& value) const
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