#pragma once

#include <emmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h> // 256, 512

#if defined(_MSC_VER)
#include <intrin.h>
#include <malloc.h>
#else
//#include <x86intrin.h>
//#include <x64intrin.h>
#include <mm_malloc.h> //
#include <cstdint>
#endif


///////////////////////////////////////////////////////////////////////////////
//

/*
 *
 */
template <typename T>
inline void AddVecLoop(const T* inArrry1, const T* inArrry2, T* inDest, std::size_t inSize)
{
    for (size_t i = 0; i < inSize; ++i)
    {
        inDest[i] = inArrry1[i] + inArrry2[i];
    }
}

template <typename T>
inline float VecDotPlain(const T* inArrry1, const T* inArrry2, std::size_t inSize)
{
    float result = 0.0f;
    for (size_t i = 0; i < inSize; ++i)
    {
        result += inArrry1[i] * inArrry2[i];
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////
// SSE


/*
 *
 * SSE implementation
 *
 */

/*
 *
 * SSE int32 unaligned
 *
 */

inline void AddVec4Int32SSEu(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest)
{
    __m128i a = _mm_loadu_si128((__m128i const*)pa);
    __m128i b = _mm_loadu_si128((__m128i const*)pb);
    __m128i c;
    c = _mm_add_epi32(a, b);
    _mm_storeu_si128((__m128i*)pdest, c);
}

inline void AddVecInt32SSEu(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        AddVec4Int32SSEu(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

inline void AddVec2Int64SSEu(const std::int64_t* pa, const std::int64_t* pb, std::int64_t* pdest)
{
    __m128i a = _mm_loadu_si128((__m128i const*)pa);
    __m128i b = _mm_loadu_si128((__m128i const*)pb);
    __m128i c;
    c = _mm_add_epi64(a, b);
    _mm_storeu_si128((__m128i*)pdest, c);
}


/*
 *
 * SSE int32 aligned
 *
 */

inline void AddVec4Int32SSE(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest)
{
    __m128i a = _mm_load_si128((__m128i const*)pa);
    __m128i b = _mm_load_si128((__m128i const*)pb);
    __m128i c;
    c = _mm_add_epi32(a, b);
    _mm_store_si128((__m128i*)pdest, c);
}

inline void AddVecInt32SSE(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        AddVec4Int32SSE(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

inline void AddVec2Int64SSE(const std::int64_t* pa, const std::int64_t* pb, std::int64_t* pdest)
{
    __m128i a = _mm_load_si128((__m128i const*)pa);
    __m128i b = _mm_load_si128((__m128i const*)pb);
    __m128i c;
    c = _mm_add_epi64(a, b);
    _mm_store_si128((__m128i*)pdest, c);
}

/*
 *
 * SSE float unaligned
 *
 */

inline void AddVec4FloatSSEu(const float* pa, const float* pb, float* pdest)
{
    __m128 va = _mm_loadu_ps(pa);
    __m128 vb = _mm_loadu_ps(pb);
    __m128 vc = _mm_add_ps(va, vb);
    _mm_storeu_ps(pdest, vc);
}

inline void AddVec2DoubleSSEu(const double* pa, const double* pb, double* pdest)
{
    __m128d va = _mm_loadu_pd(pa);
    __m128d vb = _mm_loadu_pd(pb);
    __m128d vc = _mm_add_pd(va, vb);
    _mm_storeu_pd(pdest, vc);
}

inline void AddVecFloatSSEu(const float* pa, const float* pb, float* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4)
    {
        AddVec4FloatSSEu(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

// Horizontal addition to sum up the elements in sum
inline float _mm128_reduce_add_ps(__m128 sum)
{
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    const float result = _mm_cvtss_f32(sum);
    return result;
}

// Dot product of 4 floats
inline float VecDotProduct4FloatSSEu(const float* pa, const float* pb) {
    //https://github.com/srinathv/ImproveHpc/blob/master/intel/2015-compilerSamples/C%2B%2B/intrinsic_samples/intrin_dot_sample.c
    std::size_t i = 0;
    __m128 va = _mm_loadu_ps(&pa[i]);    // Load 4 floats from a
    __m128 vb = _mm_loadu_ps(&pb[i]);    // Load 4 floats from b
    __m128 pr = _mm_mul_ps(va, vb);     // Multiply the elements
    const float result = _mm128_reduce_add_ps(pr);  // Horizontal addition to sum up the elements in sum
    return result;
}

// Function to compute dot product using SSE instructions (128-bit)
inline float VecDotRetFloatSSEu(const float* pa, const float* pb, std::size_t size)
{
    std::size_t i = 0;
    __m128 sum = _mm_setzero_ps();

    // Process elements in chunks of 16 floats (512 bits)
    for (; i + 4 <= size; i += 4)
    {
        __m128 va = _mm_loadu_ps(&pa[i]);    // Load 4 floats from a
        __m128 vb = _mm_loadu_ps(&pb[i]);    // Load 4 floats from b
        __m128 pr = _mm_mul_ps(va, vb);     // Multiply the elements
        sum = _mm_add_ps(sum, pr);          // Accumulate the products
    }

    float result = _mm128_reduce_add_ps(sum);   // Horizontal addition to sum up the elements in sum

    // Handle remaining elements
    for (; i < size; ++i)
    {
        result += pa[i] * pb[i];
    }

    return result;
}


/*
 *
 * SSE float aligned
 *
 */

inline void AddVec4FloatSSE(const float* pa, const float* pb, float* pdest)
{
    __m128 va = _mm_load_ps(pa);
    __m128 vb = _mm_load_ps(pb);
    __m128 vc = _mm_add_ps(va, vb);
    _mm_store_ps(pdest, vc);
}

inline void AddVec2DoubleSSE(const double* pa, const double* pb, double* pdest)
{
    __m128d va = _mm_load_pd(pa);
    __m128d vb = _mm_load_pd(pb);
    __m128d vc = _mm_add_pd(va, vb);
    _mm_store_pd(pdest, vc);
}

inline void AddVecFloatSSE(const float* pa, const float* pb, float* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4)
    {
        AddVec4FloatSSE(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

// Dot product of 4 floats
inline float VecDotProduct4FloatSSE(const float* pa, const float* pb) {
    //https://github.com/srinathv/ImproveHpc/blob/master/intel/2015-compilerSamples/C%2B%2B/intrinsic_samples/intrin_dot_sample.c
    std::size_t i = 0;
    __m128 va = _mm_load_ps(&pa[i]);    // Load 4 floats from a
    __m128 vb = _mm_load_ps(&pb[i]);    // Load 4 floats from b
    __m128 pr = _mm_mul_ps(va, vb);     // Multiply the elements
    const float result = _mm128_reduce_add_ps(pr);  // Horizontal addition to sum up the elements in sum
    return result;
}

// Function to compute dot product using SSE instructions (128-bit)
inline float VecDotRetFloatSSE(const float* pa, const float* pb, std::size_t size)
{
    std::size_t i = 0;
    __m128 sum = _mm_setzero_ps();

    // Process elements in chunks of 16 floats (512 bits)
    for (; i + 4 <= size; i += 4)
    {
        __m128 va = _mm_load_ps(&pa[i]);    // Load 4 floats from a
        __m128 vb = _mm_load_ps(&pb[i]);    // Load 4 floats from b
        __m128 pr = _mm_mul_ps(va, vb);     // Multiply the elements
        sum = _mm_add_ps(sum, pr);          // Accumulate the products
    }

    float result = _mm128_reduce_add_ps(sum);   // Horizontal addition to sum up the elements in sum

    // Handle remaining elements
    for (; i < size; ++i)
    {
        result += pa[i] * pb[i];
    }

    return result;
}



///////////////////////////////////////////////////////////////////////////////
// AVX

/*
 * AVX int32 unaligned
 */

inline void AddVec8Int32AVXu(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest)
{
    __m256i a = _mm256_loadu_si256((__m256i const*)pa);
    __m256i b = _mm256_loadu_si256((__m256i const*)pb);
    __m256i c = _mm256_add_epi64(a, b);
    _mm256_storeu_si256((__m256i*)pdest, c);
}

inline void AddVecInt32AVXu(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        AddVec8Int32AVXu(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

/*
 * AVX int32 aligned
 */

inline void AddVec8Int32AVX(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest)
{
    __m256i a = _mm256_load_si256((__m256i const*)pa);
    __m256i b = _mm256_load_si256((__m256i const*)pb);
    __m256i c = _mm256_add_epi64(a, b);
    _mm256_store_si256((__m256i*)pdest, c);
}

inline void AddVecInt32AVX(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        AddVec8Int32AVX(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

/*
 * AVX float unaligned
 */

 // Horizontal addition to sum up the elements in sum
inline float _mm256_reduce_add_ps(__m256 sum)
{
    // Horizontal addition to sum up the elements in sum
    __m128 vlow = _mm256_castps256_ps128(sum);        // Lower 128 bits
    __m128 vhigh = _mm256_extractf128_ps(sum, 1);     // Higher 128 bits
    vlow = _mm_add_ps(vlow, vhigh);                   // Add lower and higher parts

    // Further horizontal addition
    vlow = _mm_hadd_ps(vlow, vlow);
    vlow = _mm_hadd_ps(vlow, vlow);
    float result = _mm_cvtss_f32(vlow);
    return result;
}

// Dot product of 8 floats
inline float VecDotProduct8FloatAVXu(const float* pa, const float* pb) {
    //https://github.com/srinathv/ImproveHpc/blob/master/intel/2015-compilerSamples/C%2B%2B/intrinsic_samples/intrin_dot_sample.c
    std::size_t i = 0;
    __m256 va = _mm256_loadu_ps(&pa[i]);
    __m256 vb = _mm256_loadu_ps(&pb[i]);
    __m256 pr = _mm256_mul_ps(va, vb);          // Multiply the elements
    float result = _mm256_reduce_add_ps(pr);    // Horizontal addition to sum up the elements in sum
    return result;
}

// Function to compute dot product using AVX-512 instructions (512-bit)
float VecDotRetFloatAVXu(const float* pa, const float* pb, std::size_t size)
{
    std::size_t i = 0;
    __m512 sum = _mm512_setzero_ps();

    // Process elements in chunks of 16 floats (512 bits)
    for (; i + 16 <= size; i += 16)
    {
        __m512 va = _mm512_loadu_ps(&pa[i]);
        __m512 vb = _mm512_loadu_ps(&pb[i]);
#if 0
        __m512 pr = _mm512_mul_ps(va, vb);      // Multiply the elements
        sum = _mm512_add_ps(sum, prod);         // Accumulate the products
#else
        sum = _mm512_fmadd_ps(va, vb, sum);     //performs multiplication and vertical addition
#endif
    }

    float result = _mm512_reduce_add_ps(sum);   // Horizontal addition to sum up the elements in sum

    // Handle remaining elements
    for (; i < size; ++i)
    {
        result += pa[i] * pb[i];
    }

    return result;
}

/*
 * AVX float aligned
 */

inline void AddVec8FloatAVX(const float* pa, const float* pb, float* pdest)
{
    __m256 va = _mm256_load_ps(pa);
    __m256 vb = _mm256_load_ps(pb);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_store_ps(pdest, vc);
}

inline void AddVecFloatAVX(const float* pa, const float* pb, float* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8)
    {
        AddVec8FloatAVX(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

// Dot product of 8 floats
inline float VecDotProduct8FloatAVX(const float* pa, const float* pb) {
    //https://github.com/srinathv/ImproveHpc/blob/master/intel/2015-compilerSamples/C%2B%2B/intrinsic_samples/intrin_dot_sample.c
    std::size_t i = 0;
    __m256 va = _mm256_load_ps(&pa[i]);
    __m256 vb = _mm256_load_ps(&pb[i]);
    __m256 pr = _mm256_mul_ps(va, vb);          // Multiply the elements
    float result = _mm256_reduce_add_ps(pr);    // Horizontal addition to sum up the elements in sum
    return result;
}

// Function to compute dot product using AVX-512 instructions (512-bit)
float VecDotRetFloatAVX(const float* pa, const float* pb, std::size_t size)
{
    std::size_t i = 0;
    __m512 sum = _mm512_setzero_ps();

    // Process elements in chunks of 16 floats (512 bits)
    for (; i + 16 <= size; i += 16)
    {
        __m512 va = _mm512_load_ps(&pa[i]);
        __m512 vb = _mm512_load_ps(&pb[i]);
        #if 0
        __m512 pr = _mm512_mul_ps(va, vb);      // Multiply the elements
        sum = _mm512_add_ps(sum, prod);         // Accumulate the products
        #else
        sum = _mm512_fmadd_ps(va, vb, sum);     //performs multiplication and vertical addition
        #endif
    }

    float result = _mm512_reduce_add_ps(sum);   // Horizontal addition to sum up the elements in sum

    // Handle remaining elements
    for (; i < size; ++i)
    {
        result += pa[i] * pb[i];
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
// AVX512


/*
 * AVX512 int32 unaligned
 */
inline void AddVec16Int32AVX512u(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest)
{
    __m512i a = _mm512_loadu_si512((__m512i const*)pa);
    __m512i b = _mm512_loadu_si512((__m512i const*)pb);
    __m512i c = _mm512_add_epi64(a, b);
    _mm512_storeu_si512((__m512i*)pdest, c);
}

inline void AddVecInt32AVX512u(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16)
    {
        AddVec16Int32AVX512u(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i)
    {
        pdest[i] = pa[i] + pb[i];
    }
}

/*
 * AVX512 int32 aligned
 */
inline void AddVec16Int32AVX512(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest)
{
    __m512i a = _mm512_load_si512((__m512i const*)pa);
    __m512i b = _mm512_load_si512((__m512i const*)pb);
    __m512i c = _mm512_add_epi64(a, b);
    _mm512_store_si512((__m512i*)pdest, c);
}

inline void AddVecInt32AVX512(const std::int32_t* pa, const std::int32_t* pb, std::int32_t* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16)
    {
        AddVec16Int32AVX512(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i)
    {
        pdest[i] = pa[i] + pb[i];
    }
}


/*
 * AVX512 float aligned
 */

inline void AddVec16FloatAVX512u(const float* pa, const float* pb, float* pdest)
{
    __m512 va = _mm512_loadu_ps(pa);
    __m512 vb = _mm512_loadu_ps(pb);
    __m512 vc = _mm512_add_ps(va, vb);
    _mm512_storeu_ps(pdest, vc);
}

inline void AddVecFloatAVX512u(const float* pa, const float* pb, float* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16)
    {
        AddVec16FloatAVX512u(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

// Dot product ofr 16 floats
inline float VecDotProduct16FloatAVX512u(const float* pa, const float* pb) {
    std::size_t i = 0;
    __m512 va = _mm512_loadu_ps(&pa[i]);
    __m512 vb = _mm512_loadu_ps(&pb[i]);
    __m512 pr = _mm512_mul_ps(va, vb);          // Multiply the elements
    float result = _mm512_reduce_add_ps(pr);    // Horizontal addition to sum up the elements in sum
    return result;
}

// Function to compute dot product using AVX-512 instructions (512-bit)
float VecDotRetFloatAVX512u(const float* pa, const float* pb, std::size_t size)
{
    std::size_t i = 0;
    __m512 sum = _mm512_setzero_ps();

    // Process elements in chunks of 16 floats (512 bits)
    for (; i + 16 <= size; i += 16)
    {
        __m512 va = _mm512_loadu_ps(&pa[i]);
        __m512 vb = _mm512_loadu_ps(&pb[i]);
#if 0
        __m512 pr = _mm512_mul_ps(va, vb);      // Multiply the elements
        sum = _mm512_add_ps(sum, pr);           // Accumulate the products
#else
        sum = _mm512_fmadd_ps(va, vb, sum);     // performs multiplication and vertical addition
#endif
    }

    float result = _mm512_reduce_add_ps(sum);   // Horizontal addition to sum up the elements in sum

    // Handle remaining elements
    for (; i < size; ++i)
    {
        result += pa[i] * pb[i];
    }

    return result;
}

/*
 * AVX512 float aligned
 */

inline void AddVec16FloatAVX512(const float* pa, const float* pb, float* pdest)
{
    __m512 va = _mm512_load_ps(pa);
    __m512 vb = _mm512_load_ps(pb);
    __m512 vc = _mm512_add_ps(va, vb);
    _mm512_store_ps(pdest, vc);
}

inline void AddVecFloatAVX512(const float* pa, const float* pb, float* pdest, std::size_t size)
{
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16)
    {
        AddVec16FloatAVX512(&pa[i], &pb[i], &pdest[i]);
    }
    // Add rest items
    for (; i < size; ++i) {
        pdest[i] = pa[i] + pb[i];
    }
}

// Dot product ofr 16 floats
inline float VecDotProduct16FloatAVX512(const float* pa, const float* pb) {
    std::size_t i = 0;
    __m512 va = _mm512_load_ps(&pa[i]);
    __m512 vb = _mm512_load_ps(&pb[i]);
    __m512 pr = _mm512_mul_ps(va, vb);          // Multiply the elements
    float result = _mm512_reduce_add_ps(pr);    // Horizontal addition to sum up the elements in sum
    return result;
}

// Function to compute dot product using AVX-512 instructions (512-bit)
float VecDotRetFloatAVX512(const float* pa, const float* pb, std::size_t size)
{
    std::size_t i = 0;
    __m512 sum = _mm512_setzero_ps();

    // Process elements in chunks of 16 floats (512 bits)
    for (; i + 16 <= size; i += 16)
    {
        __m512 va = _mm512_load_ps(&pa[i]);
        __m512 vb = _mm512_load_ps(&pb[i]);
#if 0
        __m512 pr = _mm512_mul_ps(va, vb);      // Multiply the elements
        sum = _mm512_add_ps(sum, pr);           // Accumulate the products
#else
        sum = _mm512_fmadd_ps(va, vb, sum);     // performs multiplication and vertical addition
#endif
    }

    float result = _mm512_reduce_add_ps(sum);   // Horizontal addition to sum up the elements in sum

    // Handle remaining elements
    for (; i < size; ++i)
    {
        result += pa[i] * pb[i];
    }

    return result;
}

void Mat4x4MultFloatAVX512(const float* pa, const float* pb, const float* pret, std::size_t size)
{

}

///////////////////////////////////////////////////////////////////////////////
// Aligned functions
// When dealing with SSE we must make sure data is DQWORD aligned (16 bytes)
// If we deal with YMM registers, 32 bytes,
// If we deal with AVX-512 (ZMM), 64 bytes aligned

#if defined(__AVX512F__)
    //
#elif defined(__AVX2__)
    //
#elif defined(__AVX__) // _M_IX86_FP
    //
#else
    //
#endif


///////////////////////////////////////////////////////////////////////////////

