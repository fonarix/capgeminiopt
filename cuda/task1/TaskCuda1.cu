

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdio.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../common/MathTypesUtils.h"
#include "../common/OsUtilsCuda.h"

// CUDA-kernet todd vectors
__global__ void vectorAdd(const float* a, const float* b, float* c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // calculate global thread index
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

double VectorMean(const VectorFloat& elements)
{
    // Function to calculate average of
    // an array using efficient method
    double avg = 0;
    for (std::size_t i = 0; i < elements.size(); i++)
    {
        avg += (elements[i] - avg) / (i + 1); // Update avg
    }
    return avg;
}

void AddVectorsGpu(const std::size_t itemsCount)
{
    cudaError_t errRet;
    size_t size = itemsCount * sizeof(float);

    // allocater memory on host (CPU)
    VectorFloat host_a = GenRandomVector(-5, 10, itemsCount);
    VectorFloat host_b = GenRandomVector(-5, 10, itemsCount);
    VectorFloat host_c = GenRandomVector(-5, 10, itemsCount);

    // device pointers (GPU)
    float* d_a, * d_b, * d_c;

    CudaEventTimer cudaTimer;
    CTimerMilliseconds chronoTimer;
    cudaTimer.Start();

    // Allocate memory on devjice
    errRet = cudaMalloc((void**)&d_a, size);
    errRet = cudaMalloc((void**)&d_b, size);
    errRet = cudaMalloc((void**)&d_c, size);

    std::cout << std::endl << "cudaMalloc " << cudaTimer.Stop() << " milliseconds";


    // Copy data from host to device
    cudaTimer.Start();
    errRet = cudaMemcpy(d_a, host_a.data(), size, cudaMemcpyHostToDevice);
    errRet = cudaMemcpy(d_b, host_b.data(), size, cudaMemcpyHostToDevice);
    std::cout << std::endl << "cudaMemcpy " << cudaTimer.Stop() << " milliseconds";

    const int threadsPerBlock = 64;
    const int blocksPerGrid = (itemsCount + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << std::endl << "GPU add vectors:";
    cudaTimer.Start();
    chronoTimer.Start();

    // Launch CUDA-kernel
    vectorAdd <<<blocksPerGrid, threadsPerBlock>>> (d_a, d_b, d_c, itemsCount);

    std::cout << std::endl << "GPU timer: " << cudaTimer.Stop() << " milliseconds";
    std::cout << std::endl << "CPU timer: " << chronoTimer.GetElapsed().count() << " milliseconds";

    // Copy results from device to host
    errRet = cudaMemcpy(host_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    double avr = VectorMean(host_c);
    std::cout << std::endl << "GPU array average: " << avr << "";

    PrintFirstElements(host_a, host_b, host_c, 10);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on host
    host_a.clear();
    host_b.clear();
    host_c.clear();

    return ;
}

void AddVectorsCpu(const std::size_t itemsCount)
{
    VectorFloat host_a = GenRandomVector(-5, 10, itemsCount);
    VectorFloat host_b = GenRandomVector(-5, 10, itemsCount);
    VectorFloat host_c;
    host_c.reserve(itemsCount);

    std::cout << std::endl << "CPU add vectors:";
    CTimerMilliseconds chronoTimer;
    chronoTimer.Start();

    std::transform(host_a.begin(), host_a.end(), host_b.begin(), std::back_inserter(host_c), [](float a_, float b_)
        {
            return a_ + b_;
        });

    std::cout << std::endl << "CPU timer: " << chronoTimer.GetElapsed().count() << " milliseconds";
    std::cout << std::endl << "CPU array average: " <<  VectorMean(host_c) << "";

    PrintFirstElements(host_a, host_b, host_c, 10);
}


int main()
{
    const std::size_t itemsCount = 100000000;
    AddVectorsGpu(itemsCount);
    AddVectorsCpu(itemsCount);

    return 0;
}

