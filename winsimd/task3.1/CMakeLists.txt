
cmake_minimum_required (VERSION 3.29)
project (task3.1)

enable_testing()

# Check simd from cmake
#[===[
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
  message(STATUS "The compiler supports /arch:AVX")
endif()
check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  message(STATUS "The compiler supports /arch:AVX2")
endif()
check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
  message(STATUS "The compiler supports /arch:AVX512")
endif()
]===]

#Executables

set(PROJECT_APP_NAME ${PROJECT_NAME})

#-DUSE_CUDA=OFF -DUSE_EIGEN=ON -DBUILD_SHARED_LIBS=OFF -DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON -DEIGEN_VECTORIZE_AVX=ON -DEIGEN_VECTORIZE_FMA=ON
#add_definitions(-DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON)

# -march=skylake-avx512

set(UTILS_DIR "../common/")
include_directories("${UTILS_DIR}")
file(GLOB_RECURSE UTILS_FILES
    "${UTILS_DIR}/CpuFeatures.h"
    "${UTILS_DIR}/Utils.h"
    "${UTILS_DIR}/SimdRoutines.h"
)

#enable_language(C ASM_MASM)

set(SRC_LIST ./task3.1.cpp)

add_executable(${PROJECT_APP_NAME}
    "${UTILS_FILES}"
    "${SRC_LIST}"
)

target_compile_features(${PROJECT_APP_NAME} PRIVATE cxx_std_17)

if (MSVC)
    #target_compile_options(${PROJECT_APP_NAME} PRIVATE "/FA /FaListingFile.asm")
    target_compile_options(${PROJECT_APP_NAME} PRIVATE "/FA") # listing
    #target_compile_options(${PROJECT_APP_NAME} PRIVATE "/arch:AVX")
    #target_compile_options(${PROJECT_APP_NAME} PRIVATE "/arch:AVX2")
    #target_compile_options(${PROJECT_APP_NAME} PRIVATE "/arch:AVX512")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
elseif(ARCH_AMD64)
elseif(LINUX)
    #https://github.com/clickhouse/clickhouse/blob/master/cmake/cpu_features.cmake

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx -mavx2 -mavx512f")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake-avx512")


endif()







