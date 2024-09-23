
#include "CpuFeatures.h"
#include "SimdRoutines.h"
#include "Utils.h"

#include <iostream>


void RunInt32AlignedTests()
{
    std::cout << std::endl <<  __FUNCTION__ << std::endl;

    auto cpuFlags = GetCpuFeatures();
    PrintCpuFeatures(cpuFlags);

    //
    constexpr std::size_t VECTOR_SIZE = 100'000'000; // 100 millions

    // Align alloc
    constexpr std::size_t alignment = 64; // Use max align for AVX-512
    std::int32_t* a = AlignedArrayAlloc<std::int32_t>(VECTOR_SIZE, alignment);
    CheckAlign(a, alignment);
    std::int32_t* b = AlignedArrayAlloc<std::int32_t>(VECTOR_SIZE, alignment);
    CheckAlign(b, alignment);
    std::int32_t* c = AlignedArrayAlloc<std::int32_t>(VECTOR_SIZE, alignment);
    CheckAlign(c, alignment);

    if (!a || !b || !c) {
        std::cerr << "alloc error" << std::endl;
        return ;
    }

    FillArrayArithmeticProgression(a, VECTOR_SIZE);
    FillArrayArithmeticProgression(b, VECTOR_SIZE, 32);

    // Number of repeats
    const int NUM_REPEATS = 50;

    double time_loop = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecLoop<std::int32_t>, a, b, c, VECTOR_SIZE);
    std::cout << "Time loop......: " << time_loop << " seconds" << std::endl;

    // Performance SSE
    double time_sse = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32SSE, a, b, c, VECTOR_SIZE);
    std::cout << "Time SSE.......: " << time_sse << " seconds" << std::endl;

    // Performance AVX
    double time_avx = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32AVX, a, b, c, VECTOR_SIZE);
    std::cout << "Time AVX.......: " << time_avx << " seconds" << std::endl;

    // Performance AVX-512
    double time_avx512 = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecInt32AVX512, a, b, c, VECTOR_SIZE);
    std::cout << "Time AVX-512...: " << time_avx512 << " seconds" << std::endl;

    //
    std::cout << std::endl << "First elements: ";

    PrintArray(a, 18);
    PrintArray(b, 18);
    PrintArray(c, 18);
    std::cout << std::endl;

    //
    AlignedArrayFree(a);
    AlignedArrayFree(b);
    AlignedArrayFree(c);
}

int main()
{
    RunInt32AlignedTests();
    return 0;
}

