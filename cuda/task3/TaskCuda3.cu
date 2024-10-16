
/* Practical Task 3: Reduction Operations Using CUDA
 * Objective: Implement a reduction operation to compute the sum of elements in an array using
 * CUDA. Explore different optimization techniques such as parallel reduction and warp divergence
 * handling
 *
 * Links:
 * https://www.youtube.com/watch?v=bpbit8SPMxU
 */

#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <vector>
#include "../common/MathTypesUtils.h"
#include "../common/OsUtilsCuda.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

//*
__global__ void sumKernel(float* input, float* output, int n)
{
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    // Copy data from global memory to shared memory
    if (i < n)
        sum = input[i];
    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];

    sharedData[tid] = sum;
    __syncthreads();

    // Sum reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // zero thread save result to output array
    if (tid == 0)
        output[blockIdx.x] = sharedData[0];
}

void MeasureReductionGpu(const VectorFloat& arr)
{
    const size_t N = arr.size();
    const size_t size = N * sizeof(float);

    // Device handles
    float* d_input, * d_intermediate, * d_temp;

    // Calculate blocks in grid and number of threads in block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on device
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_intermediate, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_temp, blocksPerGrid * sizeof(float));

    //
    cudaMemcpy(d_input, arr.data(), size, cudaMemcpyHostToDevice);

    CudaEventTimer cudaTimer;
    cudaTimer.Start();

    // Run first kernel to calc partial sums
    sumKernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (d_input, d_intermediate, N);

    int s = blocksPerGrid;
    while (s > 1)
    {
        int threads = (s > threadsPerBlock) ? threadsPerBlock : s;
        int blocks = (s + threads - 1) / threads;

        sumKernel << <blocks, threads, threads * sizeof(float) >> > (d_intermediate, d_intermediate, s);

        // switch front and back buffers
        s = blocks;
    }

    // Copy result:
    float sumOfElements = 0.0f;
    cudaMemcpy(&sumOfElements, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::endl << "GPU Timer: " << cudaTimer.Stop() << " milliseconds";
    std::cout << std::endl << "Sum Of Elements: " << sumOfElements;

    // Free resources
    cudaFree(d_input);
    cudaFree(d_intermediate);
    cudaFree(d_temp);

    return ;
}

void MeasureReductionGpuThrust(const VectorFloat& arr)
{
    std::cout << std::endl << "Runing thrust::reduce... " << arr.size() << " elements";

    thrust::device_vector<float> d_vec = arr;
    CudaEventTimer cudaTimer;
    cudaTimer.Start();

    float sumOfElements = thrust::reduce(thrust::device, d_vec.begin(), d_vec.end());

    std::cout << std::endl << "GPU Timer: " << cudaTimer.Stop() << " milliseconds";
    std::cout << std::endl << "Sum Of Elements: " << sumOfElements;
}

void MeasureReductionCpu(const VectorFloat& arr)
{
    std::cout << std::endl << "Runing std::accumulate... " << arr.size() << " elements";
    CTimerMicroseconds chronoTimer;
    chronoTimer.Start();

    float sumOfElements = std::accumulate(arr.begin(), arr.end(), VectorFloat::value_type(0));

    std::cout << std::endl << "CPU timer: " << chronoTimer.GetElapsed().count() << " microseconds";
    std::cout << std::endl << "Sum Of Elements: " << sumOfElements;
}

int main()
{
    const std::size_t itemsCount = 50000000;// 100000000;
    VectorFloat arrayGpu1  = GenRandomVector(-10, 10, itemsCount);
    VectorFloat arrayGpu2 = GenRandomVector(-10, 10, itemsCount);
    VectorFloat arrayCpu   = GenRandomVector(-10, 10, itemsCount);

    MeasureReductionGpu(arrayGpu1);
    PrintFirstArrayElements(arrayGpu1.data(), 5);

    MeasureReductionGpuThrust(arrayGpu2);
    PrintFirstArrayElements(arrayGpu2.data(), 5);

    MeasureReductionCpu(arrayCpu);
    PrintFirstArrayElements(arrayCpu.data(), 5);

    return 0;
}

