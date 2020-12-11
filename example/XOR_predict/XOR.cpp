
#include <iostream>
#include <string>

#include "../../neural_network/neural_network.hpp"

int main()
{
    // Make layers early
    layers MyLayers = layers(4);
    MyLayers.in(2);
    MyLayers.in(4, ACT_SIGMOID);
    MyLayers.in(4, ACT_SIGMOID);
    MyLayers.in(1, ACT_TANH);
    NeuralNetwork MyNN = NeuralNetwork(MyLayers);

    // Datasets
    MATRIX(float) xs = { {1,0},{0,1},{0,0},{1,1} };
    MATRIX(float) ys = { {1},{1},{0},{0} };

    Datasets datasets("example/XOR_predict/XOR.csv");
    MyNN.train(datasets, {"x","y"},{"output"}, 10000);


    // Training data without csv file
    // MyNN.train(xs, ys, 10000);

    // Predict data
    std::vector<float> predict_value0 = MyNN.predict(xs[0]);
    std::vector<float> predict_value1 = MyNN.predict(xs[1]);
    std::vector<float> predict_value2 = MyNN.predict(xs[2]);
    std::vector<float> predict_value3 = MyNN.predict(xs[3]);

    // print outputs
    for (float k : predict_value0)
        std::cout << k << std::endl;
    for (float k : predict_value1)
        std::cout << k << std::endl;
    for (float k : predict_value2)
        std::cout << k << std::endl;
    for (float k : predict_value3)
        std::cout << k << std::endl;

    return 0;
}