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
    /*
    template<class T>
    std::string FormatWithCommas(T value)
    {
        std::stringstream ss;
        // otherwise, use "en_US.UTF-8" as the locale name
        ss.imbue(std::locale("ko_KR.UTF-8"));
        ss << std::fixed << value;
        return ss.str();
    }

    inline void PrintElapsed()//const char*prefix, const char* postfix)
    {
        const char* prefix = "Elapsed: ";
        const char* postfix = " [ns]";
        std::cout
            << std::endl
            << prefix
            << std::setfill(' ') << std::setw(16)
            << FormatWithCommas(GetElapsed().count())
            << postfix;
    }
    */
};

//using CTimerNanoseconds = TTimer<std::chrono::nanoseconds>;
using CTimerMicroseconds = TTimer<std::chrono::microseconds>;