#pragma once

#include <chrono>
#include <iostream>
#include <vector>
#include <random>

using VectorFloat = std::vector<float>;
VectorFloat GenRandomVector(float minValue, float maxValue, std::size_t inCount);

void PrintFirstElements(const VectorFloat& a, const VectorFloat& b, const VectorFloat& c, int n);


template<class T>
void PrintFirstArrayElements(const T* a, std::size_t count)
{
    std::cout << std::endl << "First " << count << " elements:\n";
    if (count > 0)
    {
        std::cout << a[0];
    }
    for (std::size_t i = 1; i < count; i++)
    {
        std::cout << ", " << a[i];
    }
    std::cout << std::endl;
}
