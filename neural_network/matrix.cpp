#include <time.h>

#include "matrix.hpp"

Matrix::Matrix(short rows, short cols) 
	:rows(rows), cols(cols)
{
    srand((unsigned int)time(NULL));
    for (short i = 0; i < rows; i++)
    {
        std::vector<float> cols_data;
        for (short j = 0; j < cols; j++)
            cols_data.push_back(0.0f);
        arr2d.push_back(cols_data);
    }
}

template<typename T>
void Matrix::SetMat(T arr[]) {
    unsigned short arr_index = 0;
    for (short i = 0; i < rows; i++)
    {
        for (short j = 0; j < cols; j++)
        {
            arr2d[i][j] = arr[arr_index];
            arr_index++;
        }
    }
}

template<typename T>
void Matrix::SetMat(std::vector<T> arr) {
    unsigned short arr_index = 0;
    for (short i = 0; i < rows; i++)
    {
        for (short j = 0; j < cols; j++)
        {
            arr2d[i][j] = arr[arr_index];
            arr_index++;
        }
    }
}

void Matrix::SetRandom() {
    for (short i = 0; i < rows; i++)
        for (short j = 0; j < cols; j++)
            arr2d[i][j] = (rand() % 100) * 0.01f;
}

void Matrix::add(std::vector<std::vector<float>> add) {
    for (short i = 0; i < rows; i++)
        for (short j = 0; j < cols; j++)
            arr2d[i][j] = arr2d[i][j] + add[i][j];
}

std::vector<std::vector<float>> Matrix::multiplyArray2d(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b)
{
    std::vector<std::vector<float>> result;
    for (short i = 0; i < a.size(); i++) {
        std::vector<float> cols;
        for (short j = 0; j < b[0].size(); j++) {
            float sum = 0;
            for (short k = 0; k < a[0].size(); k++) 
                sum += a[i][k] * b[k][j];
            cols.push_back(sum);
        }
        result.push_back(cols);
    }
    return result;
}

std::vector<std::vector<float>> Matrix::array2d(std::vector<float> array)
{
    std::vector<std::vector<float>> result;
    for (short i = 0; i < array.size(); i++)
    {
        std::vector<float> rows;
        rows.push_back(array[i]);
        result.push_back(rows);
    }
    return result;
}

std::vector<std::vector<float>> Matrix::transpose(std::vector<std::vector<float>> array)
{
    float** temporary_array2d;
    temporary_array2d = new float* [array[0].size()];
    for (short i = 0; i < array[0].size(); i++)
        temporary_array2d[i] = new float[array.size()];

    for (int i = 0; i < array.size(); i++)
        for (int j = 0; j < array[i].size(); j++)
            temporary_array2d[j][i] = array[i][j];

    // temporary_array to <vector>
    std::vector<std::vector<float>> result;
    for (short i = 0; i < array[0].size(); i++)
    {
        std::vector<float> rows;
        for (short j = 0; j < array.size(); j++)
        {
            rows.push_back(temporary_array2d[i][j]);
        }
        result.push_back(rows);
    }

    // delete temporary_array2d
    for (short i = 0; i < array[0].size(); i++)
            delete[] temporary_array2d[i];
    delete[] temporary_array2d;

    return result;
}
