
#include "CpuFeatures.h"
#include "SimdRoutines.h"
#include "Utils.h"

#include <iostream>


void RunAddFloatAlignedTests()
{
    std::cout << std::endl << __FUNCTION__ << std::endl;

    auto cpuFlags = GetCpuFeatures();
    PrintCpuFeatures(cpuFlags);

    //
    constexpr std::size_t VECTOR_SIZE = 100'000'000; // 100 millions

    // Align alloc
    constexpr std::size_t alignment = 64; // Use max align for AVX-512
    float* a = AlignedArrayAlloc<float>(VECTOR_SIZE, alignment);
    CheckAlign(a, alignment);
    float* b = AlignedArrayAlloc<float>(VECTOR_SIZE, alignment);
    CheckAlign(b, alignment);
    float* c = AlignedArrayAlloc<float>(VECTOR_SIZE, alignment);
    CheckAlign(c, alignment);

    if (!a || !b || !c) {
        std::cerr << "alloc error" << std::endl;
        return;
    }

    FillArrayArithmeticProgression(a, VECTOR_SIZE, 0.2f);
    FillArrayArithmeticProgression(b, VECTOR_SIZE, 32.2f);

    // Number of repeats
    const int NUM_REPEATS = 50;

    double time_loop = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecLoop<float>, a, b, c, VECTOR_SIZE);
    std::cout << "Time loop......: " << time_loop << " seconds" << std::endl;

    // Performance SSE
    double time_sse = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecFloatSSE, a, b, c, VECTOR_SIZE);
    std::cout << "Time SSE.......: " << time_sse << " seconds" << std::endl;

    // Performance AVX
    double time_avx = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecFloatAVX, a, b, c, VECTOR_SIZE);
    std::cout << "Time AVX.......: " << time_avx << " seconds" << std::endl;

    // Performance AVX-512
    double time_avx512 = MeasureTimeFunc2VecAdd<NUM_REPEATS>(AddVecFloatAVX512, a, b, c, VECTOR_SIZE);
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


void RunDotProductUnalignedTests()
{
    std::cout << std::endl << __FUNCTION__ << std::endl;

    //constexpr std::size_t VECTOR_SIZE = 10'000'000; // 10 millions
    constexpr std::size_t VECTOR_SIZE = 1'000'000; // 1 million

    // Align alloc
    constexpr std::size_t alignment = 32;
    float* a = ArrayAlloc<float>(VECTOR_SIZE);
    CheckAlign(a, alignment);
    float* b = ArrayAlloc<float>(VECTOR_SIZE);
    CheckAlign(b, alignment);

    if (!a || !b) {
        std::cerr << "alloc error" << std::endl;
        return;
    }

    FillArrayArithmeticProgression(a, VECTOR_SIZE, 50.2f, 0.1f);
    FillArrayArithmeticProgression(b, VECTOR_SIZE, 10.2f, 0.1f);

    // Number of repeats
    const int NUM_REPEATS = 5000;

    float  dot1 = 0.0f;
    double time_loop = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotPlain<float>, a, b, &dot1, VECTOR_SIZE);
    std::cout << "Time loop......: " << time_loop << " seconds" << std::endl;

    // Performance SSE
    float  dot2 = 0.0f;
    double time_sse = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotRetFloatSSEu, a, b, &dot2, VECTOR_SIZE);
    std::cout << "Time SSE.......: " << time_sse << " seconds" << std::endl;

    // Performance AVX
    float  dot3 = 0.0f;
    double time_avx = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotRetFloatAVXu, a, b, &dot3, VECTOR_SIZE);
    std::cout << "Time AVX.......: " << time_avx << " seconds" << std::endl;

    // Performance AVX-512
    float  dot4 = 0.0f;
    double time_avx512 = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotRetFloatAVX512u, a, b, &dot4, VECTOR_SIZE);
    std::cout << "Time AVX-512...: " << time_avx512 << " seconds" << std::endl;

    //
    std::cout << std::endl << "First elements: ";

    PrintArray(a, 18);
    PrintArray(b, 18);
    //PrintArray(c, 18);
    std::cout << std::endl << std::endl
        << std::setprecision(10)
        << "Dot products:   " << std::endl
        << "Plain..: " << dot1 << std::endl
        << "SSE....: " << dot2 << std::endl
        << "AVX....: " << dot3 << std::endl
        << "AVX512.: " << dot4;
    std::cout << std::endl;

    //
    ArrayFree(a);
    ArrayFree(b);
}


void RunDotProductAlignedTests()
{
    std::cout << std::endl << __FUNCTION__ << std::endl;

    //constexpr std::size_t VECTOR_SIZE = 10'000'000; // 10 millions
    constexpr std::size_t VECTOR_SIZE = 1'000'000; // 1 million

    // Align alloc
    constexpr std::size_t alignment = 64; // Use max align for AVX-512
    float* a = AlignedArrayAlloc<float>(VECTOR_SIZE, alignment);
    CheckAlign(a, alignment);
    float* b = AlignedArrayAlloc<float>(VECTOR_SIZE, alignment);
    CheckAlign(b, alignment);

    if (!a || !b) {
        std::cerr << "alloc error" << std::endl;
        return;
    }

    FillArrayArithmeticProgression(a, VECTOR_SIZE, 50.2f, 0.1f);
    FillArrayArithmeticProgression(b, VECTOR_SIZE, 10.2f, 0.1f);

    // Number of repeats
    const int NUM_REPEATS = 5000;

    float  dot1 = 0.0f;
    double time_loop = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotPlain<float>, a, b, &dot1, VECTOR_SIZE);
    std::cout << "Time loop......: " << time_loop << " seconds" << std::endl;

    // Performance SSE
    float  dot2 = 0.0f;
    double time_sse = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotRetFloatSSE, a, b, &dot2, VECTOR_SIZE);
    std::cout << "Time SSE.......: " << time_sse << " seconds" << std::endl;

    // Performance AVX
    float  dot3 = 0.0f;
    double time_avx = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotRetFloatAVX, a, b, &dot3, VECTOR_SIZE);
    std::cout << "Time AVX.......: " << time_avx << " seconds" << std::endl;

    // Performance AVX-512
    float  dot4 = 0.0f;
    double time_avx512 = MeasureTimeFunc2VecDot<NUM_REPEATS>(VecDotRetFloatAVX512, a, b, &dot4, VECTOR_SIZE);
    std::cout << "Time AVX-512...: " << time_avx512 << " seconds" << std::endl;

    //
    std::cout << std::endl << "First elements: ";

    PrintArray(a, 18);
    PrintArray(b, 18);
    //PrintArray(c, 18);
    std::cout << std::endl << std::endl
        << std::setprecision(10)
        << "Dot products:   " << std::endl
        << "Plain..: " << dot1 << std::endl
        << "SSE....: " << dot2 << std::endl
        << "AVX....: " << dot3 << std::endl
        << "AVX512.: " << dot4;
    std::cout << std::endl;

    //
    AlignedArrayFree(a);
    AlignedArrayFree(b);
}

int main()
{
    RunAddFloatAlignedTests();
    RunDotProductUnalignedTests();
    RunDotProductAlignedTests();
    return 0;
}

