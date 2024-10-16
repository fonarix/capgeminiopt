# build
cmake_minimum_required (VERSION 3.20)
project("cudatasks" LANGUAGES CXX CUDA)
#cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#links
#https://developer.nvidia.com/blog/building-cuda-applications-cmake/
#https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html

#include(CTest)
enable_testing()
set(CMAKE_CXX_STANDART 17)
set(CMAKE_CXX_STANDART_REQUIRED ON)


#[[
include(ExternalProject)

ExternalProject_Add(googletest
        GIT_REPOSITORY    https://github.com/google/googletest.git
        GIT_TAG           v1.15.2
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/googletest-src"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/googletest-build"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )

ExternalProject_Add(benchmark
        GIT_REPOSITORY    https://github.com/google/benchmark.git
        GIT_TAG           v1.9.0
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/benchmark-src"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/benchmark-build"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )

#Thrust
#[[
ExternalProject_Add(Thrust
        GIT_REPOSITORY    https://github.com/NVIDIA/thrust.git
        GIT_TAG           v2.1.0
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Thrust-src"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/Thrust-build"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )
]]

# https://stackoverflow.com/questions/75168735/how-to-link-cuda-dynamic-libraries-using-cmake

#MathTypesUtils.h MathTypesUtils.cpp

file(GLOB COMMON_UTILS_SRCS
    ./common/OsUtilsCuda.h ./common/OsUtilsCuda.cpp
    ./common/MathTypesUtils.h ./common/MathTypesUtils.cpp
)

add_executable(TaskCuda1 ./task1/TaskCuda1.cu ${COMMON_UTILS_SRCS})
set_property(TARGET TaskCuda1 PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#matrix_mul_cublas.cu
#TaskCuda2.cu
#add_executable(TaskCuda2 ./task2/matrix_mul_cublas.cu ${COMMON_UTILS_SRCS} task2/MulMatrixCuda.cuh task2/MulMatrixCuda.cu)
#set_property(TARGET TaskCuda2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#find_package(cuBLAS REQUIRED CONFIG)
#target_link_libraries(TaskCuda2 cuBLAS)

#
add_executable(TaskCuda3 ./task3/TaskCuda3.cu ${COMMON_UTILS_SRCS})
set_property(TARGET TaskCuda3 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#find_package(Thrust REQUIRED CONFIG)
#thrust_create_target(Thrust)
#target_link_libraries(TaskCuda3 Thrust)
#

#
add_executable(TaskCuda4 ./task4/TaskCuda4.cu ${COMMON_UTILS_SRCS})
set_property(TARGET TaskCuda4 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
target_link_libraries(TaskCuda4 Thrust)


#target_link_libraries(particle_test PRIVATE particles)

#if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
#  set_property(TARGET particle_test
#             PROPERTY
#           BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#endif()


#
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE})
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})


#set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_APP_NAME})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "TaskCuda1")



