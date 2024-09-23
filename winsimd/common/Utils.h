#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <array>


/*
 * Returns incrmental array
 */
template <typename T, std::size_t size>
inline
std::array<T, size> GenArrayArithmeticProgression(T inStartValue = 0, T inStepValue = 1)
{
    std::array<T, size> ret;

    for (size_t i = 0; i < size; ++i)
    {
        ret[i] = inStartValue + i*inStepValue;
    }

    return ret; //rvo/Copy elision
}

template <typename T>
inline
void FillArrayArithmeticProgression(T* ioArray, std::size_t inItemsCount, T inStartValue = 0, T inStepValue = 1)
{
    for (std::size_t i = 0; i < inItemsCount; ++i)
    {
        ioArray[i] = inStartValue + i * inStepValue;
    }

    return;
}

template <typename T, std::size_t size>
void PrintArray(const T(&array)[size])
{
    for (size_t i = 0; i < size; ++i)
        std::cout << array[i] << " ";
}

template <typename T>
void PrintArray(const T *inArray, std::size_t inItemsCount)
{
    std::cout << std::endl << std::setfill(' ');
    for (size_t i = 0; i < inItemsCount; ++i)
    {
        std::cout << std::setw(4) << inArray[i];
    }
}

template <>
void PrintArray(const float* inArray, std::size_t inItemsCount)
{
    std::cout << std::endl << std::setfill(' ') << std::setprecision(4);
    for (std::size_t i = 0; i < inItemsCount; ++i)
    {
        //std::setprecision(std::numeric_limits<double>::digits10 + 1)
        //std::cout << std::setw(4) << std::scientific << inArray[i];
        std::cout << std::setw(6) << std::showpoint << inArray[i];
    }
}

template<class T>
std::string FormatWithCommas(T value)
{
    std::stringstream ss;
    // otherwise, use "en_US.UTF-8" as the locale name
    ss.imbue(std::locale("ko_KR.UTF-8"));
    ss << std::fixed << value;
    return ss.str();
}

template<typename TDurationType = std::chrono::nanoseconds>
class TTimer
{
    using clock_type = std::chrono::high_resolution_clock;// steady_clock;
    using time_point = clock_type::time_point;
    //using TDuration = std::chrono::duration<std::chrono::nanoseconds>;
    time_point mBegin;

public:
    TTimer()
    {
        Start();
    }

    inline void Start()
    {
        mBegin = clock_type::now();
    }

    inline TDurationType GetElapsed()
    {
        const time_point end = clock_type::now();
        //const std::chrono::duration<TDuration> elapsed = end - mBegin;
        //const TDuration elapsed = end - mBegin;
        const auto elapsed = std::chrono::duration_cast<TDurationType> (end - mBegin);

        return elapsed;
    }

    inline void PrintElapsed()//const char*prefix, const char* postfix)
    {
        const char* prefix = "Elapsed: ";
        const char* postfix = " [ns]"; // "[�s]"
        std::cout
            << std::endl
            << prefix //"Time difference = "
            << std::setfill(' ') << std::setw(16)
            << FormatWithCommas(GetElapsed().count())
            << postfix;
    }
};

//using CTimerNanoseconds = TTimer<double>;
using CTimerNanoseconds = TTimer<std::chrono::nanoseconds>;
using CTimerMicroseconds = TTimer<std::chrono::microseconds>;


// measure execution time numRepeats times, and return avetage value
template <std::size_t numRepeats, typename Func, typename T>
double MeasureTimeFunc2VecAdd(Func func, const T* a, const T* b, T* c, std::size_t size) {
    double totalTime = 0.0;
    for (size_t i = 0; i < numRepeats; ++i)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        func(a, b, c, size);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        totalTime += elapsed.count();
    }
    return totalTime / numRepeats; // return average
}
// measure execution time numRepeats times, and return avetage value
template <std::size_t numRepeats, typename Func, typename T>
double MeasureTimeFunc2VecDot(Func func, const T* a, const T* b, T* c, std::size_t size) {
    double totalTime = 0.0;
    for (std::size_t i = 0; i < numRepeats; ++i)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        *c = func(a, b, size);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        totalTime += elapsed.count();
    }
    return totalTime / numRepeats; // return average
}

void* AlignedAlloc(std::size_t inSize, std::size_t inAlignment)
{
#if _WIN32
    //return _mm_malloc(inSize, inAlignment);
    return _aligned_malloc(inSize, inAlignment);
#elif __linux__
    void* memPtr = nullptr;
    int result = posix_memalign(&memPtr, inAlignment, inSize);
    if (EINVAL == result)
    {
        return nullptr;
    }
    if (ENOMEM == result)
    {
        return nullptr;
    }
    return memPtr;
#endif
}

template<class T>
T* AlignedArrayAlloc(std::size_t inCount, std::size_t inAlignment)
{
    const std::size_t size = inCount * sizeof(T);
#if defined(_MSC_VER)
    return static_cast<T*>(_aligned_malloc(size, inAlignment));
#else
    return static_cast<T*>(aligned_alloc(size, inAlignment));
#endif
}

bool IsAligned(const void* inPtr, std::size_t inAlignment)
{
    const std::size_t address = reinterpret_cast<size_t>(inPtr);
    const std::size_t rest = address % inAlignment;
    if (0 == rest)
    {
        return true;
    }
    return false;
}

template<typename T>
void CheckAlign(T* ptr, std::size_t align)
{
    std::cout << "Pointer " << ptr;
    if (IsAligned(ptr, align))
    {
        std::cout << " aligned " << align << " well" << std::endl;
        return;
    }
    std::cout << " " << align << " NOT aligned! " << std::endl;
    return;
}

void AlignedArrayFree(void* inPtr)
{
#if _WIN32
    //_mm_free(inPtr);
    _aligned_free(inPtr);
#elif __linux__
    free(inPtr); // free mem after posix_memalign
#endif
}

template<class T>
T* ArrayAlloc(size_t inCount)
{
    const std::size_t size = inCount * sizeof(T);
    return static_cast<T*>(malloc(size));
}

void ArrayFree(void* inPtr)
{
    free(inPtr);
}


