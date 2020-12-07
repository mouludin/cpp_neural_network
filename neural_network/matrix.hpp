// My own matrix library for neural network

#pragma once

#include <cstdlib>
#include <vector>

class Matrix
{
public:
    short rows;
    short cols;
    std::vector<std::vector<float>> arr2d;

public:
    Matrix(short rows, short cols);

    template<typename T>
    void SetMat(T arr[]);

    template<typename T>
    void SetMat(std::vector<T> arr);

    void SetRandom();

    void add(std::vector<std::vector<float>> add);

    static std::vector<std::vector<float>> multiplyArray2d(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b);
    static std::vector<std::vector<float>> array2d(std::vector<float> array);
    static std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> array);
};
