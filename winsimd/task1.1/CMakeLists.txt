
cmake_minimum_required (VERSION 3.29)
project (task1.1)

enable_testing()

#Executables

set(PROJECT_APP_NAME "task1.1")
#set(PROJECT_APP_NAME ${PROJECT_NAME})

#-DUSE_CUDA=OFF -DUSE_EIGEN=ON -DBUILD_SHARED_LIBS=OFF -DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON -DEIGEN_VECTORIZE_AVX=ON -DEIGEN_VECTORIZE_FMA=ON
add_definitions(-DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

#set(CURRENT_DIR ".")
#set(HEADERS_DIR "${CURRENT_DIR}/") #include
set(UTILS_DIR "../common/") #include
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
endif()

