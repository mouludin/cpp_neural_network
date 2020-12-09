// My own matrix library for neural network
#pragma once

#include <cstdlib>
#include <vector>

// Datatype
#define MATRIX(X) std::vector<std::vector<X>>

class Matrix
{
public:
    short rows;
    short cols;
    MATRIX(float) arr2d;

public:
    Matrix(short rows, short cols);

    template<typename T>
    void SetMat(T arr[]);

    template<typename T>
    void SetMat(std::vector<T> arr);

    void SetRandom();

    void add(MATRIX(float) add);

    static MATRIX(float) multiplyMatrix(MATRIX(float) a, MATRIX(float) b);
    static MATRIX(float) tomatrix(std::vector<float> array);
    static MATRIX(float) transpose(MATRIX(float) array);
};
