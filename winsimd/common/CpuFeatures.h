#ifndef __CpuFeatures_h__
#define __CpuFeatures_h__

#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#else
//#include <x86intrin.h>
//#include <stdio.h>
//#include <mm_malloc.h>
#endif



///////////////////////////////////////////////////////////////////////////////
// CPU features

const unsigned      kCpuFeatureSSE = 1;
const unsigned      kCpuFeatureSSE2 = 2;
const unsigned      kCpuFeatureSSE3 = 4;
const unsigned      kCpuFeatureSSSE3 = 8;    // Supplemental Streaming SIMD Extensions 3 (SSSE3 or SSE3S) is a SIMD instruction set created by Intel and is the fourth iteration of the SSE technology.
const unsigned      kCpuFeatureSSE41 = 16;
const unsigned      kCpuFeatureSSE42 = 32;
const unsigned      kCpuFeatureAVX = 64;
const unsigned      kCpuFeatureAVX2 = 128;
const unsigned      kCpuFeatureAVX512 = 256; // https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set

#if defined(__GNUC__) || defined(__clang__)

inline unsigned GetCpuFeatures()
{
    unsigned resultFlags = 0;
    __builtin_cpu_init();
//    https://gcc.gnu.org/onlinedocs/gcc-4.9.2/gcc/X86-Built-in-Functions.html
    if (__builtin_cpu_supports("sse"))
    {
        resultFlags |= kCpuFeatureSSE;
    }
    if (__builtin_cpu_supports("sse2"))
    {
        resultFlags |= kCpuFeatureSSE2;
    }
    if (__builtin_cpu_supports("sse3"))
    {
        resultFlags |= kCpuFeatureSSE3;
    }
    if (__builtin_cpu_supports("ssse3"))
    {
        resultFlags |= kCpuFeatureSSSE3;
    }
    if (__builtin_cpu_supports("sse4.1"))
    {
        resultFlags |= kCpuFeatureSSE41;
    }
    if (__builtin_cpu_supports("sse4.2"))
    {
        resultFlags |= kCpuFeatureSSE42;
    }
    if (__builtin_cpu_supports("avx"))
    {
        resultFlags |= kCpuFeatureAVX;
    }
    if (__builtin_cpu_supports("avx2"))
    {
        resultFlags |= kCpuFeatureAVX2;
    }
    if (__builtin_cpu_supports("avx512f")) // https://stackoverflow.com/questions/6121792/how-to-check-if-a-cpu-supports-the-sse3-instruction-set
    {
        resultFlags |= kCpuFeatureAVX512;
    }

    if (__builtin_cpu_is("amd"))
    {
        //std::cout << "amd" << std::endl;
    }
    if (__builtin_cpu_is("corei7"))
    {
        //std::cout << "corei7" << std::endl;
    }

    return resultFlags;
}

#elif _MSC_VER

inline unsigned GetCpuFeatures()
{
    unsigned resultFlags = 0;

    // Get cpu max function id
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int maxFunctionId = cpuInfo[0];

    // check basic instructions
    if (maxFunctionId >= 1)
    {
        __cpuid(cpuInfo, 1);
        const bool sse = cpuInfo[3] & (1 << 25);
        const bool sse2 = cpuInfo[3] & (1 << 26);
        const bool sse3 = cpuInfo[2] & (1 << 0);
        const bool ssse3 = cpuInfo[2] & (1 << 9);
        const bool sse41 = cpuInfo[2] & (1 << 19);
        const bool sse42 = cpuInfo[2] & (1 << 20);
        const bool avx = cpuInfo[2] & (1 << 28);

        // Check support OS AVX
        const bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27);
        const unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        const bool avxSupported = avx && osUsesXSAVE_XRSTORE && (xcrFeatureMask & 0x6);

        resultFlags |= sse ? kCpuFeatureSSE : 0;
        resultFlags |= sse2 ? kCpuFeatureSSE2 : 0;
        resultFlags |= sse3 ? kCpuFeatureSSE3 : 0;
        resultFlags |= ssse3 ? kCpuFeatureSSSE3 : 0;
        resultFlags |= sse41 ? kCpuFeatureSSE41 : 0;
        resultFlags |= sse42 ? kCpuFeatureSSE42 : 0;
        resultFlags |= avxSupported ? kCpuFeatureAVX : 0;
    }

    // Check extended features
    if (maxFunctionId >= 7) {
        __cpuid(cpuInfo, 7);
        const bool avx2 = cpuInfo[1] & (1 << 5);
        const bool avx512f = cpuInfo[1] & (1 << 16);

        resultFlags |= avx2 ? kCpuFeatureAVX2 : 0;
        resultFlags |= avx512f ? kCpuFeatureAVX512 : 0;
    }

    // Check extended features
    if (maxFunctionId >= 16) {
        __cpuid(cpuInfo, 7);
        //const bool avx512f = cpuInfo[1] & (1 << 16);
    }
    return resultFlags;
}
#endif

inline const char* CpuFeatureFlagToString(unsigned flag)
{
    if (flag == kCpuFeatureSSE)   { return "SSE";    }
    if (flag == kCpuFeatureSSE2)  { return "SSE2";   }
    if (flag == kCpuFeatureSSE3)  { return "SSE3";   }
    if (flag == kCpuFeatureSSSE3) { return "SSSE3";  }
    if (flag == kCpuFeatureSSE41) { return "SSE41";  }
    if (flag == kCpuFeatureSSE42) { return "SSE42";  }
    if (flag == kCpuFeatureAVX)   { return "AVX";    }
    if (flag == kCpuFeatureAVX2)  { return "AVX2";   }
    if (flag == kCpuFeatureAVX512){ return "AVX512"; }
    return "";
}

inline void PrintCpuFeatures(unsigned flags)
{
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureSSE)    << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureSSE2)   << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureSSE3)   << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureSSSE3)  << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureSSE41)  << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureSSE42)  << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureAVX)    << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureAVX2)   << std::endl;
    std::cout << CpuFeatureFlagToString(flags & kCpuFeatureAVX512) << std::endl;
}


#endif //__CpuFeatures_h__
