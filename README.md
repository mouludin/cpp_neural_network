# Mouludin - C++ Neural Network

Minimalist neural network library for machine learning &amp; deep learning in C++

## Getting started


### Example XOR problem

### Include this library and vector library
```c++
  #include <vector>

  #include "neural_network.h"
```

#### Determine the layers:

```c++
    // Make layers early
    layers MyLayers = layers(4);
    MyLayers.in(2);
    MyLayers.in(4, ACT_SIGMOID);
    MyLayers.in(4, ACT_SIGMOID);
    MyLayers.in(1, ACT_TANH);

    NeuralNetwork MyNN = NeuralNetwork(MyLayers);
```
You can give more than 4 layers. it depends on how much you need.
Note: minimum is 3 layers

I have provided some activation functions that you can use:

  * ACT_SIGMOID as sigmoid function (Range = (0,1))

  * ACT_TANH as hyperbolic tangent / tanh function (Range = (-1,1))


#### Training data:

```c++
    // DATASETS
    // input
    std::vector<std::vector<float>> xs = { {1,0},{0,1},{0,0},{1,1} };
    // output
    std::vector<std::vector<float>> ys = { {1},{1},{0},{0} };

    // Training data or Calculate error
    MyNN.train(xs, ys, 10000);
```
Note: the greater the number of epohcs. the smaller errors you get


#### Prediction:
```c++
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
```


#### Output:
```c++
0.993184
0.993164
0.00360237
0.00491454
```

*Note: Help me improve this library.

## that is a simple example of using this library


## Authors

* **Muhammad Mauludin Anwar** - *Initial work* - [mouludin](https://github.com/mouludin)

## License

This project is licensed under the terms of the MIT license, see LICENSE.
