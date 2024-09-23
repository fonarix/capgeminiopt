
cmake_minimum_required (VERSION 3.29)
project (task1.1)

enable_testing()

#Executables

set(PROJECT_APP_NAME "task1.1")
#set(PROJECT_APP_NAME ${PROJECT_NAME})

if (MSVC)
#-DUSE_CUDA=OFF -DUSE_EIGEN=ON -DBUILD_SHARED_LIBS=OFF -DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON -DEIGEN_VECTORIZE_AVX=ON -DEIGEN_VECTORIZE_FMA=ON
add_definitions(-DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON)
elseif(LINUX)
    #DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL
endif()

#https://github.com/clickhouse/clickhouse/blob/master/cmake/cpu_features.cmake

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

#set(CURRENT_DIR ".")
#set(HEADERS_DIR "${CURRENT_DIR}/") #include
set(UTILS_DIR "./../common/") #include
include_directories("${UTILS_DIR}")
file(GLOB_RECURSE UTILS_FILES
    "${UTILS_DIR}/CpuFeatures.h"
    "${UTILS_DIR}/Utils.h"
    "${UTILS_DIR}/SimdRoutines.h"
)

#enable_language(C ASM_MASM)

###################################################
#includes

#set(SRC_LIST ./task1.cpp ./Utils.h ./SimdRoutines.h)
set(SRC_LIST ./task1.cpp)

add_executable(${PROJECT_APP_NAME}
    "${UTILS_FILES}"
    "${SRC_LIST}"
)


target_compile_features(${PROJECT_APP_NAME} PRIVATE cxx_std_17)


if (MSVC)
    #target_compile_options(${PROJECT_APP_NAME} PRIVATE "/Fa /FaListingFile.asm")
    #target_compile_options(${PROJECT_APP_NAME} PRIVATE "/Fa" )
    target_compile_options(${PROJECT_APP_NAME} PRIVATE /FA)

elseif(ARCH_AMD64)

elseif(LINUX)
    #https://github.com/clickhouse/clickhouse/blob/master/cmake/cpu_features.cmake

    option (ENABLE_SSSE3 "Use SSSE3 instructions on x86_64" 1)
    option (ENABLE_SSE41 "Use SSE4.1 instructions on x86_64" 1)
    option (ENABLE_SSE42 "Use SSE4.2 instructions on x86_64" 1)
    option (ENABLE_PCLMULQDQ "Use pclmulqdq instructions on x86_64" 1)
    option (ENABLE_POPCNT "Use popcnt instructions on x86_64" 1)
    option (ENABLE_AVX "Use AVX instructions on x86_64" 1)
    option (ENABLE_AVX2 "Use AVX2 instructions on x86_64" 1)
    option (ENABLE_AVX512 "Use AVX512 instructions on x86_64" 1)
    #option (ENABLE_AVX512_VBMI "Use AVX512_VBMI instruction on x86_64 (depends on ENABLE_AVX512)" 0)
    #option (ENABLE_BMI "Use BMI instructions on x86_64" 0)
    #option (ENABLE_BMI2 "Use BMI2 instructions on x86_64 (depends on ENABLE_AVX2)" 0)
    #option (ENABLE_AVX2_FOR_SPEC_OP "Use avx2 instructions for specific operations on x86_64" 0)
    option (ENABLE_AVX512_FOR_SPEC_OP "Use avx512 instructions for specific operations on x86_64" 1)

    if (ENABLE_SSSE3)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -mssse3")
    endif ()

    if (ENABLE_SSE41)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -msse4.1")
    endif ()

    if (ENABLE_SSE42)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -msse4.2")
    endif ()


    if (ENABLE_AVX)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -mavx")
    endif ()

    if (ENABLE_AVX2)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -mavx2")
        if (ENABLE_BMI2)
            set (COMPILER_FLAGS "${COMPILER_FLAGS} -mbmi2")
        endif ()
    endif ()

    if (ENABLE_AVX512)
        set (COMPILER_FLAGS "${COMPILER_FLAGS} -mavx512f -mavx512bw -mavx512vl")
        if (ENABLE_AVX512_VBMI)
            set (COMPILER_FLAGS "${COMPILER_FLAGS} -mavx512vbmi")
        endif ()
    endif ()

    if (ENABLE_AVX512_FOR_SPEC_OP)
        set (X86_INTRINSICS_FLAGS "-mbmi -mavx512f -mavx512bw -mavx512vl -mprefer-vector-width=256")
    endif ()


    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -mavx2 -mavx512f")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake-avx512")


endif()

