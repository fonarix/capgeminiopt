// task1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

/* prefs
/arch:AVX512
*/

#include "Utils.h"
#include "SimdRoutines.h"

#include <iostream>
//#include <chrono>
//#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h> // 256, 512
#include <functional>
#include <array>



void runtest128()
{
    static const std::int64_t cRepeats = 1000'000'000llu;

    CTimerMicroseconds timer;

    std::cout << "\n\n\nSimd test xmm 128bit\n";

    int intSeq1[4] = { 0,1,2,3 };
    int intSeq2[4] = { 4,5,6,7 };
    int intDest[4] = { 0,0,6,0 };

    timer.Start();
    for (auto i = 0; i < cRepeats; i++)
    {
        AddVecInt32SSEu(intSeq1, intSeq2, intDest, 4);
        memset(&intDest, 0, sizeof(intDest));
    }
    timer.PrintElapsed();
    PrintArray(intDest);

    //
    std::cout << "\nLoop test\n";

    timer.Start();
    for (auto i = 0; i < cRepeats; i++)
    {
        AddVecLoop<int>(intSeq1, intSeq2, intDest, 4);
        memset(&intDest, 0, sizeof(intDest)); // avoid optimization
    }
    timer.PrintElapsed();
    PrintArray(intDest);
}


void runtest256()
{
    static const std::int64_t cRepeats = 1000'000'000llu;

    CTimerNanoseconds timer;

    std::cout << "\n\n\nSimd test ymmx AVX2\n";

    int intSeq1[8] = { 0,1,2,3,4,5,6,7 };
    int intSeq2[8] = { 4,5,6,7,8,9,10,11 };
    int intDest[8] = { 0,0,0,0,0,0,0,0 };

    timer.Start();
    for (auto i = 0; i < cRepeats; i++)
    {
        AddVecInt32AVXu(intSeq1, intSeq2, intDest, 8);
        memset(&intDest, 0, sizeof(intDest));
    }
    timer.PrintElapsed();
    PrintArray(intDest);

    //
    std::cout << "\nLoop test\n";

    timer.Start();
    for (auto i = 0; i < cRepeats; i++)
    {
        AddVecLoop<int>(intSeq1, intSeq2, intDest, 8);
        memset(&intDest, 0, sizeof(intDest)); // avoid optimization
    }
    timer.PrintElapsed();
    PrintArray(intDest);
}

void runtest512()
{
    static const std::int64_t cRepeats = 1000'000'000llu;

    CTimerNanoseconds timer;

    std::cout << "\n\n\nSimd test zmm AVX512\n";

    int intSeq1[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 };
    int intSeq2[16] = { 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19 };
    int intDest[16] = { 0 };

    timer.Start();
    for (auto i = 0; i < cRepeats; i++)
    {
        AddVecInt32AVX512u(intSeq1, intSeq2, intDest, 16);
        //memset(&intDest, 0, sizeof(intDest));
    }
    timer.PrintElapsed();
    PrintArray(intDest);

    //
    std::cout << "\n\nLoop test\n";

    timer.Start();
    for (auto i = 0; i < cRepeats; i++)
    {
        AddVecLoop<int>(intSeq1, intSeq2, intDest, 16);
        //Add16int32Loop(intSeq1, intSeq2, intDest);
        //memset(&intDest, 0, sizeof(intDest)); // avoid optimization
        //memset(&intDest, 0, sizeof(intDest));
    }
    timer.PrintElapsed();

    PrintArray(intDest);
}


/*
void runtest2()
{

    CTimerNanoseconds timer;
    //timer.PrintElapsed();

    std::cout << "\nSimd test\n";
    int my_int_sequence[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    //int intSeq[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    //int intSeq[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };

    int intSeq1[4] = { 0,1,2,3 };
    int intSeq2[4] = { 4,5,6,7 };
    int intDest[4] = { 0,0,6,0 };

    timer.Start();
    //
    //AddVecInt32SSEu(intSeq1, intSeq2, intDest);
    //CallRepeater2([&]() { AddVecInt32SSEu(intSeq1, intSeq2, intDest); });
    CallRepeater2([&]() { AddVecInt32SSEu(intSeq1, intSeq2, intDest); memset(&intDest, 0, sizeof(intDest)); });
    //std::array<char, 3> arr;
    //CallRepeater<4>();
    intDest[0] = intDest[1] + intDest[2]; // tepm operation avoid optimization
    timer.PrintElapsed();
    std::cout << "\nLoop test\n";

    // new test
    timer.Start();
    //AddVecLoop<int>(intSeq1, intSeq2, intDest, 4);
    //Add4int32Loop(intSeq1, intSeq2, intDest);
    //CallRepeater2([&]() { Add4int32Loop(intSeq1, intSeq2, intDest);});
    CallRepeater2([&]() { Add4int32Loop(intSeq1, intSeq2, intDest); memset(&intDest, 0, sizeof(intDest)); });
    timer.PrintElapsed();
}
//*/

int main()
{
    auto arr1 = GenArrayArithmeticProgression<int, 10240>();
    auto arr2 = GenArrayArithmeticProgression<int, 16>(11);


    runtest128();
    runtest256();
    runtest512();

    CTimerNanoseconds timer;
    timer.Start();
    std::cout << "Hello World2!\n";
    timer.PrintElapsed();

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
