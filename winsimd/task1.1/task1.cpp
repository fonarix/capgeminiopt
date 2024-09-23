// task1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CpuFeatures.h"
#include "SimdRoutines.h"
#include "Utils.h"

#include <iostream>
#include <vector>

void RunWithStdVector()
{
    auto cpuFlags = GetCpuFeatures();
    PrintCpuFeatures(cpuFlags);
    //
    constexpr size_t VECTOR_SIZE = 100'000'000; // 100 millions

    // Align alloc
    constexpr size_t alignment = 16;
    auto a = std::vector<int>(VECTOR_SIZE);
    CheckAlign(a.data(), alignment);
    auto b = std::vector<int>(VECTOR_SIZE);
    CheckAlign(b.data(), alignment);
    auto c = std::vector<int>(VECTOR_SIZE);
    CheckAlign(c.data(), alignment);

    FillArrayArithmeticProgression(a.data(), VECTOR_SIZE);
    FillArrayArithmeticProgression(b.data(), VECTOR_SIZE, 32);

    // Number of repeats
    constexpr int NUM_REPEATS = 10;

    double timeLoop = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecLoop<int>, a.data(), b.data(), c.data(), VECTOR_SIZE);
    std::cout << "Time loop......: " << timeLoop << " seconds" << std::endl;

    // Performance SSE
    double time_sse = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32SSEu, a.data(), b.data(), c.data(), VECTOR_SIZE);
    std::cout << "Time SSE.......: " << time_sse << " seconds" << std::endl;

    // Performance AVX
    double time_avx = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32AVXu, a.data(), b.data(), c.data(), VECTOR_SIZE);
    std::cout << "Time AVX.......: " << time_avx << " seconds" << std::endl;

    // Performance AVX-512
    double time_avx512 = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32AVX512u, a.data(), b.data(), c.data(), VECTOR_SIZE);
    std::cout << "Time AVX-512...: " << time_avx512 << " seconds" << std::endl;

    //
    std::cout << std::endl << "First resulting elements: ";

    PrintArray(a.data(), 18);
    PrintArray(b.data(), 18);
    PrintArray(c.data(), 18);
    std::cout << std::endl;

    // free
    a.clear();
    b.clear();
    c.clear();
}

void Run()
{
    auto cpuFlags = GetCpuFeatures();
    PrintCpuFeatures(cpuFlags);

    //
    constexpr size_t VECTOR_SIZE = 100'000'000; // 100 millions

    // Align alloc
    constexpr size_t alignment = 16; // Use max align for AVX-512
    int* a = ArrayAlloc<int>(VECTOR_SIZE);
    CheckAlign(a, alignment);
    int* b = ArrayAlloc<int>(VECTOR_SIZE);
    CheckAlign(b, alignment);
    int* c = ArrayAlloc<int>(VECTOR_SIZE);
    CheckAlign(c, alignment);

    if (!a || !b || !c) {
        std::cerr << "alloc error" << std::endl;
        return ;
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
}

int main()
{
    RunWithStdVector();
    return 0;
}

