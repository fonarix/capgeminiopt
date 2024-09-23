// task1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CpuFeatures.h"
#include "SimdRoutines.h"
#include "Utils.h"

#include <iostream>
//#include <chrono>
//#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h> // 256, 512
#include <functional>
#include <array>

int main()
{

    auto cpuFlags = GetCpuFeatures();
    PrintCpuFeatures(cpuFlags);

    //
    constexpr std::size_t VECTOR_SIZE = 100'000'000; // 100 millions

    // Align alloc
    constexpr std::size_t alignment = 16; // Use max align for AVX-512
    int* a = ArrayAlloc<int>(VECTOR_SIZE);
    CheckAlign(a, alignment);
    int* b = ArrayAlloc<int>(VECTOR_SIZE);
    CheckAlign(b, alignment);
    int* c = ArrayAlloc<int>(VECTOR_SIZE);
    CheckAlign(c, alignment);

    if (!a || !b || !c) {
        std::cerr << "alloc error" << std::endl;
        return 1;
    }

    FillArrayArithmeticProgression(a, VECTOR_SIZE);
    FillArrayArithmeticProgression(b, VECTOR_SIZE, 32);

    // Number of repeats
    constexpr int NUM_REPEATS = 10;

    double timeLoop = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecLoop<int>, a, b, c, VECTOR_SIZE);
    std::cout << "Time loop......: " << timeLoop << " seconds" << std::endl;

    // Performance SSE
    double time_sse = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32SSEu, a, b, c, VECTOR_SIZE);
    std::cout << "Time SSE.......: " << time_sse << " seconds" << std::endl;

    // Performance AVX
    double time_avx = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32AVXu, a, b, c, VECTOR_SIZE);
    std::cout << "Time AVX.......: " << time_avx << " seconds" << std::endl;

    // Performance AVX-512
    double time_avx512 = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32AVX512u, a, b, c, VECTOR_SIZE);
    std::cout << "Time AVX-512...: " << time_avx512 << " seconds" << std::endl;

    //
    std::cout << std::endl << "First resulting elements: ";

    PrintArray(a, 18);
    PrintArray(b, 18);
    PrintArray(c, 18);
    std::cout << std::endl;

    // free
    ArrayFree(a);
    ArrayFree(b);
    ArrayFree(c);
    return 0;
}

