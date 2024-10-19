
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

/*
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
//*/



__device__ void warpReduce(volatile float* shmem_ptr, int t) {
    shmem_ptr[t] += shmem_ptr[t + 32];
    shmem_ptr[t] += shmem_ptr[t + 16];
    shmem_ptr[t] += shmem_ptr[t + 8];
    shmem_ptr[t] += shmem_ptr[t + 4];
    shmem_ptr[t] += shmem_ptr[t + 2];
    shmem_ptr[t] += shmem_ptr[t + 1];
}

__global__ void sum_reduction(float* v, float* v_r) {
    // Allocate shared memory
    extern __shared__ float partial_sum[];

    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements AND do first add of reduction
    // Vector now 2x as long as number of threads, so scale i
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Store first partial result instead of just the elements
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
    __syncthreads();

    // Start at 1/2 block stride and divide by two each iteration
    // Stop early (call device function instead)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Let the thread 0 for this block write it's result to main memory
    // Result is inexed by this block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

void MeasureReductionGpu(const VectorFloat& arr)
{
    // Vector size
    int n = arr.size();
    size_t bytes = n * sizeof(int);

    // Original vector and result vector
    //float* h_v, * h_v_r;
    float* d_v, * d_v_r;

    // Allocate memory
    //h_v = (float*)malloc(bytes);
    float* h_v_r = (float*)malloc(bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    // Copy to device
    cudaMemcpy(d_v, arr.data(), bytes, cudaMemcpyHostToDevice);

    // TB Size
    int TB_SIZE = 256;

    // Grid Size (cut in half) (No padding)
    int GRID_SIZE = n / TB_SIZE / 2;

    // Call kernel
    sum_reduction << <GRID_SIZE, TB_SIZE, TB_SIZE * sizeof(float) >> > (d_v, d_v_r);

    sum_reduction << <1, TB_SIZE, TB_SIZE * sizeof(float) >> > (d_v_r, d_v_r);

    // Copy to host;
    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Accumulated result is %d \n", h_v_r[0]);
    //scanf("Press enter to continue: ");
    //assert(h_v_r[0] == 65536);

    printf("COMPLETED SUCCESSFULLY\n");
}

void MeasureReductionGpuTmp(const VectorFloat& arr)
{
    const size_t N = arr.size();
    const size_t size = N * sizeof(float);

    // Device handles
    float* d_input, * d_intermediate, * d_temp;

    // Calculate blocks in grid and number of threads in block
    int threadsPerBlock = 256;
    //int blocksPerGrid = N / threadsPerBlock / 2;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //int blocksPerGrid = (N + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

    // Allocate memory on device
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_intermediate, blocksPerGrid * sizeof(float));
    cudaMalloc((void**)&d_temp, blocksPerGrid * sizeof(float));

    //
    cudaMemcpy(d_input, arr.data(), size, cudaMemcpyHostToDevice);

    CudaEventTimer cudaTimer;
    cudaTimer.Start();

    // Call kernel
    sum_reduction << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (d_input, d_intermediate);
    sum_reduction << <1, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (d_intermediate, d_intermediate);

    // Copy result:
    float sumOfElements = 0.0f;
    //if (input != d_intermediate)
    //{
    //    cudaMemcpy(&sumOfElements, input, sizeof(float), cudaMemcpyDeviceToHost);
    //}
    //else
    //{
    cudaMemcpy(&sumOfElements, d_intermediate, sizeof(float), cudaMemcpyDeviceToHost);
    //}

    std::cout << std::endl << "GPU Timer: " << cudaTimer.Stop() << " milliseconds";
    std::cout << std::endl << "Sum Of Elements: " << sumOfElements;

    // Free resources
    cudaFree(d_input);
    cudaFree(d_intermediate);
    cudaFree(d_temp);

    return;
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
    const std::size_t itemsCount = 10;// 50000000;// 100000000;
    VectorFloat arrayGpu1  = GenRandomVector(-10, 10, 10);//itemsCount);
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

