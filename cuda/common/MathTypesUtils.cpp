


#include "MathTypesUtils.h"
#include <iomanip>
#include <iostream>
//#include <sstream>
#include <random>


namespace {

template<typename T = int, char sep = ' ', int size>
std::string arrayToStr(T(&inArray)[size])
{
    std::stringstream stream;
    stream << std::dec;
    if (size > 0)
    {
        stream << std::dec << inArray[0];
    }

    for (auto i = 1; i < size; i++)
    {
        stream << std::dec << sep << inArray[i];
    }

    return stream.str();
}

}

//https://stackoverflow.com/questions/42525713/cuda-how-to-sum-all-elements-of-an-array-into-one-number-within-the-gpu
//https://stackoverflow.com/questions/22939034/block-reduction-in-cuda

VectorFloat GenRandomVector(float minValue, float maxValue, std::size_t inCount)
{
    VectorFloat vec;
    vec.reserve(inCount);
    vec.resize(inCount);

    // std::mt19937 is the goto PRNG in <random>
    // This declaration also seeds the PRNG using std::random_device
    // A std::uniform_int_distribution is exactly what it sounds like
    // Every number in the range is equally likely to occur.

    //std::seed_seq seq{ 1, 2, 3, 4, 5 };
    //std::seed_seq seq{ 42 };
    std::mt19937 prng(42);// same random seed for testing
    //std::mt19937 prng(std::random_device{}());
    //std::uniform_int_distribution<int> dist(minValue, maxValue);
    std::uniform_real_distribution<float> dist(minValue, maxValue);

    for (auto& i : vec)
    {
        i = dist(prng);
    }

    return vec;
}

void PrintFirstElements(const VectorFloat& a, const VectorFloat& b, const VectorFloat& c, int count)
{
    std::cout << std::endl << "First " << count << " elements:\n";
    for (std::size_t i = 0; i < count; i++)
    {
        std::cout << a[i] << " + " << b[i] << " = " << std::fixed << std::setprecision(8) << c[i] << std::endl;
    }
}



// https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu

/*
*********************************************************************
description: dot product of two matrix (not only square) in CPU,
             for validating GPU results

parameters:
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C)
            to store the result
return: none
*********************************************************************
*/
void matrix_mult_cpu(int* h_a, int* h_b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}