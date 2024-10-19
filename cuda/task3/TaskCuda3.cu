//
#include <algorithm>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <thread>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

#include "../common/OsUtilsCuda.h"

#include "TaskCuda3Utils.h"


// single version
// partial_sum size same size as threadsPerBlock * sizeof(float)
__global__ void sum_reduction_o(float* v, float* v_r, int n)
{
    extern __shared__ float partial_sum[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy to shared memory
    if (i < n)
    {
        partial_sum[threadIdx.x] = v[i];
    }
    else
    {
        partial_sum[threadIdx.x] = 0.0f;  // write null in case out of index
    }
    __syncthreads();

    // Reduction inside shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write reduction result of every block to global memory
    if (threadIdx.x == 0)
    {
        v_r[blockIdx.x] = partial_sum[0];
    }
}


// Warp reduction routine (32 threads)
__device__ void warpReduce(volatile float* shmem_ptr, int t)
{
    shmem_ptr[t] += shmem_ptr[t + 32];
    shmem_ptr[t] += shmem_ptr[t + 16];
    shmem_ptr[t] += shmem_ptr[t + 8];
    shmem_ptr[t] += shmem_ptr[t + 4];
    shmem_ptr[t] += shmem_ptr[t + 2];
    shmem_ptr[t] += shmem_ptr[t + 1];
}

//
// Kernel reducts x2 input buffer
// Fetches x2 twice more data from global memory
__global__ void sum_reduction2(float* v, float* v_r, int n)
{
    extern __shared__ float partial_sum[];

    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0;

    // Check memory limits
    if (i              < n) sum += v[i];
    if (i + blockDim.x < n) sum += v[i + blockDim.x];

    // Copy to shared memory
    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction inside shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    // Final reduction for last 32 threads, in last warp
    if (threadIdx.x < 32)
    {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Write reduction result of every block to global memory
    if (threadIdx.x == 0)
    {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

// Kernel reducts x3 input buffer
__global__ void sum_reduction3(float* v, float* v_r, int n)
{
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 3) + threadIdx.x;

    float sum = 0.0f;

    // Check memory limits
    if (i                  < n) sum  = v[i];
    if (i +     blockDim.x < n) sum += v[i + blockDim.x];
    if (i + 2 * blockDim.x < n) sum += v[i + 2 * blockDim.x];

    // Save result to shared memory
    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    // Редукция внутри блока
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    // Final reduction for last 32 threads, in warp
    if (threadIdx.x < 32)
    {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Write reduction result of every block to global memory
    if (threadIdx.x == 0)
    {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

// Kernel reducts x4 input buffer
__global__ void sum_reduction4(float* v, float* v_r, int n) {
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;

    float sum = 0.0f;

    // Check memory limits
#if 1
    //if (i < n)
    {
        //sum = v[i];
        if (i                  < n) sum  = v[i];
        #if 1
        if (i +     blockDim.x < n) sum += v[i +     blockDim.x];
        if (i + 2 * blockDim.x < n) sum += v[i + 2 * blockDim.x];
        if (i + 3 * blockDim.x < n) sum += v[i + 3 * blockDim.x];
        #else
        const int blockDimx2 = blockDim.x + blockDim.x;
        const int blockDimx3 = blockDimx2 + blockDim.x;

#if 0
        if (i + blockDim.x < n) sum += v[i + blockDim.x];
        if (i + blockDimx2 < n) sum += v[i + blockDimx2];
        if (i + blockDimx3 < n) sum += v[i + blockDimx3];
#else
        if (i + blockDimx3 < n)
        {
            sum += v[i + blockDim.x] + v[i + blockDimx2] + v[i + blockDimx3];
        }
        else
        if (i + blockDimx2 < n)
        {
            sum += v[i + blockDim.x] + v[i + blockDimx2];
        }
        else
        if (i + blockDim.x < n)
        {
            sum += v[i + blockDim.x];
        }
#endif

        #endif
    }
#else
    const int blockDimx2 = blockDim.x + blockDim.x;
    const int blockDimx3 = blockDimx2 + blockDim.x;

    if (i + blockDimx3 < n)
    {
        //sum = v[i] + v[i + blockDim.x] + v[i + 2 * blockDim.x] + v[i + 3 * blockDim.x];
        sum = v[i] + v[i + blockDim.x] + v[i + blockDimx2] + v[i + blockDimx3];
    }
    else if (i + blockDimx2 < n)
    {
        //sum = v[i] + v[i + blockDim.x] + v[i + 2 * blockDim.x];
        const int blockDimx2 = blockDim.x + blockDim.x;
        sum = v[i] + v[i + blockDim.x] + v[i + blockDimx2];
    }
    else if (i + blockDim.x < n)
    {
        sum = v[i] + v[i + blockDim.x];
    }
    else if (i < n)
    {
        sum = v[i];
    }
#endif

    // Save result to shared memory
    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction inside shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    //__syncthreads();
    // Final reduction for last 32 threads, in warp
    if (threadIdx.x < 32)
    {
        warpReduce(partial_sum, threadIdx.x);
    }

    __syncthreads();

    // Write reduction result of every block to global memory
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

// Kernel reducts x6 input buffer
__global__ void sum_reduction6(float* v, float* v_r, int n) {
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 6) + threadIdx.x;

    float sum = 0.0f;
    // Check memory limits
    if (i < n) {
        sum = v[i];
#if 0
        if (i + blockDim.x < n) sum += v[i + blockDim.x];
        if (i + 2 * blockDim.x < n) sum += v[i + 2 * blockDim.x];
        if (i + 3 * blockDim.x < n) sum += v[i + 3 * blockDim.x];
        if (i + 4 * blockDim.x < n) sum += v[i + 4 * blockDim.x];
        if (i + 5 * blockDim.x < n) sum += v[i + 5 * blockDim.x];
#else
        const int blockDimx2 = blockDim.x + blockDim.x;
        const int blockDimx3 = blockDimx2 + blockDim.x;
        const int blockDimx4 = blockDimx3 + blockDim.x;
        const int blockDimx5 = blockDimx4 + blockDim.x;

        if (i + blockDim.x < n) sum += v[i + blockDim.x];
        if (i + blockDimx2 < n) sum += v[i + blockDimx2];
        if (i + blockDimx3 < n) sum += v[i + blockDimx3];
        if (i + blockDimx4 < n) sum += v[i + blockDimx4];
        if (i + blockDimx5 < n) sum += v[i + blockDimx5];

#endif
    }

    // Save result to shared memory
    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction inside shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final reduction for last 32 threads, in warp
    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Write reduction result of every block to global memory
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

__global__ void sum_reduction10(float* v, float* v_r, int n)
{
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 10) + threadIdx.x;

    // Check memory limits
    float sum = 0.0f;
    //if (i < n)
    {
        //sum = v[i];
        if (i                  < n) sum  = v[i];
#if 1
        if (i +     blockDim.x < n) sum += v[i +     blockDim.x];
        if (i + 2 * blockDim.x < n) sum += v[i + 2 * blockDim.x];
        if (i + 3 * blockDim.x < n) sum += v[i + 3 * blockDim.x];
        if (i + 4 * blockDim.x < n) sum += v[i + 4 * blockDim.x];
        if (i + 5 * blockDim.x < n) sum += v[i + 5 * blockDim.x];
        if (i + 6 * blockDim.x < n) sum += v[i + 6 * blockDim.x];
        if (i + 7 * blockDim.x < n) sum += v[i + 7 * blockDim.x];
        if (i + 8 * blockDim.x < n) sum += v[i + 8 * blockDim.x];
        if (i + 9 * blockDim.x < n) sum += v[i + 9 * blockDim.x];
#else
        const int blockDimx2 = blockDim.x + blockDim.x;
        const int blockDimx3 = blockDimx2 + blockDim.x;
        const int blockDimx4 = blockDimx3 + blockDim.x;
        const int blockDimx5 = blockDimx4 + blockDim.x;

        if (i + blockDim.x < n) sum += v[i + blockDim.x];
        if (i + blockDimx2 < n) sum += v[i + blockDimx2];
        if (i + blockDimx3 < n) sum += v[i + blockDimx3];
        if (i + blockDimx4 < n) sum += v[i + blockDimx4];
        if (i + blockDimx5 < n) sum += v[i + blockDimx5];

#endif
    }

    // Save result to shared memory
    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    // Редукция внутри блока
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Последняя редукция для варпов
    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Записываем результат блока в глобальную память
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

__global__ void sum_reduction20(float* v, float* v_r, int n)
{
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 20) + threadIdx.x;

    // Check memory limits
    float sum = 0.0f;
    //if (i < n)
    {
        //sum = v[i];
        if (i                   < n) sum  = v[i];
#if 1
        if (i +      blockDim.x < n) sum += v[i +      blockDim.x];
        if (i +  2 * blockDim.x < n) sum += v[i +  2 * blockDim.x];
        if (i +  3 * blockDim.x < n) sum += v[i +  3 * blockDim.x];
        if (i +  4 * blockDim.x < n) sum += v[i +  4 * blockDim.x];
        if (i +  5 * blockDim.x < n) sum += v[i +  5 * blockDim.x];
        if (i +  6 * blockDim.x < n) sum += v[i +  6 * blockDim.x];
        if (i +  7 * blockDim.x < n) sum += v[i +  7 * blockDim.x];
        if (i +  8 * blockDim.x < n) sum += v[i +  8 * blockDim.x];
        if (i +  9 * blockDim.x < n) sum += v[i +  9 * blockDim.x];
        if (i + 10 * blockDim.x < n) sum += v[i + 10 * blockDim.x];
        if (i + 11 * blockDim.x < n) sum += v[i + 11 * blockDim.x];
        if (i + 12 * blockDim.x < n) sum += v[i + 12 * blockDim.x];
        if (i + 13 * blockDim.x < n) sum += v[i + 13 * blockDim.x];
        if (i + 14 * blockDim.x < n) sum += v[i + 14 * blockDim.x];
        if (i + 15 * blockDim.x < n) sum += v[i + 15 * blockDim.x];
        //*
        if (i + 16 * blockDim.x < n) sum += v[i + 16 * blockDim.x];
        if (i + 17 * blockDim.x < n) sum += v[i + 17 * blockDim.x];
        if (i + 18 * blockDim.x < n) sum += v[i + 18 * blockDim.x];
        if (i + 19 * blockDim.x < n) sum += v[i + 19 * blockDim.x];
        //if (i + 20 * blockDim.x < n) sum += v[i + 20 * blockDim.x];
        //if (i + 21 * blockDim.x < n) sum += v[i + 21 * blockDim.x];
        //*/
#else
        const int blockDimx2 = blockDim.x + blockDim.x;
        const int blockDimx3 = blockDimx2 + blockDim.x;
        const int blockDimx4 = blockDimx3 + blockDim.x;
        const int blockDimx5 = blockDimx4 + blockDim.x;

        if (i + blockDim.x < n) sum += v[i + blockDim.x];
        if (i + blockDimx2 < n) sum += v[i + blockDimx2];
        if (i + blockDimx3 < n) sum += v[i + blockDimx3];
        if (i + blockDimx4 < n) sum += v[i + blockDimx4];
        if (i + blockDimx5 < n) sum += v[i + blockDimx5];

#endif
    }

    // Save result to shared memory
    partial_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction inside shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final reduction for last 32 threads, in warp
    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Write reduction result of every block to global memory
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

// Host routine to launch reduction kernels
ReduceFuncResult reduce1(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 1;

    // Round up blocksCount count
    //int blocksCount = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    //int blocksCount = (n + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu1", threadsPerBlock, REDUCE_MULTIPLIER);

    CudaEventTimer totalTimer;
    totalTimer.Start();

    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction_o << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    //FinalReduceOnGpu(blocksCount, threadsPerBlock, devVecResult);
    while (blocksCount > 1)
    {
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

        sum_reduction_o << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();

    //std::cout << std::endl << "GPU timer: " << measure.mReduceTimeMs << " milliseconds";

    // For debug:
    // intermediate results
    //std::vector<float> h_v_r;
    //h_v_r.resize(blocksCount);
    // Copy intermediate results to host
    //cudaMemcpy(h_v_r.data(), devVecResult, blocksCount * sizeof(float), cudaMemcpyDeviceToHost);

    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}

// Host routine to launch reduction kernels
ReduceFuncResult reduce2(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 2;
    // Round up blocksCount count
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu2", threadsPerBlock, REDUCE_MULTIPLIER);
    CudaEventTimer totalTimer;
    totalTimer.Start();

    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction2 << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    while (blocksCount > 1)
    {
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        //cudaDeviceSynchronize();
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

        sum_reduction2 << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}

// Host routine to launch reduction kernels
ReduceFuncResult reduce3(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 3;
    // Round up blocksCount count
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu3", threadsPerBlock, REDUCE_MULTIPLIER);
    CudaEventTimer totalTimer;
    totalTimer.Start();

    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction3 << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    while (blocksCount > 1)
    {
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);
        //std::this_thread::sleep_for(std::chrono::milliseconds(500));
        //cudaDeviceSynchronize();
        sum_reduction3 << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}

// Host routine to launch reduction kernels
ReduceFuncResult reduce4(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 4;
    // Round up blocksCount count
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu4", threadsPerBlock, REDUCE_MULTIPLIER);
    CudaEventTimer totalTimer;
    totalTimer.Start();

    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction4 << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    //FinalReduceOnGpu(blocksCount, threadsPerBlock, devVecResult);
    while (blocksCount > 1)
    {
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

        sum_reduction4 << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}


// Host routine to launch reduction kernels
ReduceFuncResult reduce6(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 6;
    // Round up blocksCount count
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu6", threadsPerBlock, REDUCE_MULTIPLIER);
    CudaEventTimer totalTimer;
    totalTimer.Start();
    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction6 << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    while (blocksCount > 1)
    {
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

        sum_reduction6 << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}

// Host routine to launch reduction kernels
ReduceFuncResult reduce10(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 10;
    // Round up blocksCount count
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu10", threadsPerBlock, REDUCE_MULTIPLIER);
    CudaEventTimer totalTimer;
    totalTimer.Start();
    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction10 << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    //FinalReduceOnGpu(blocksCount, threadsPerBlock, devVecResult);
    while (blocksCount > 1)
    {
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

        sum_reduction10 << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}

// Host routine to launch reduction kernels
ReduceFuncResult reduce20(const std::vector<float>& hostVec, int threadsPerBlock)
{
    //std::cout << std::endl << __FUNCTION__;
    const std::size_t n = hostVec.size();
    const int REDUCE_MULTIPLIER = 20;
    // Round up blocksCount count
    int blocksCount = (n + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

    ReduceFuncResult measure("redu20", threadsPerBlock, REDUCE_MULTIPLIER);
    CudaEventTimer totalTimer;
    totalTimer.Start();
    CudaEventTimer timer;
    timer.Start();

    // Allocate device memory
    float* devVec = nullptr;
    float* devVecResult = nullptr;
    cudaMalloc(&devVec, n * sizeof(float));
    cudaMalloc(&devVecResult, blocksCount * sizeof(float));

    measure.mMallocTimeMs = timer.Stop();
    timer.Start();

    // Copy data to device
    cudaMemcpy(devVec, hostVec.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    // Launch reduction kernel
    sum_reduction20 << <blocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVec, devVecResult, n);

    while (blocksCount > 1)
    {
        int nextBlocksCount = (blocksCount + threadsPerBlock * REDUCE_MULTIPLIER - 1) / (threadsPerBlock * REDUCE_MULTIPLIER);

        sum_reduction20 << <nextBlocksCount, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (devVecResult, devVecResult, blocksCount);
        blocksCount = nextBlocksCount; // Update blocksCount count for next reduction step
    }

    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    float sum = 0.0f;
    cudaMemcpy(&sum, devVecResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);

    measure.mCopyDeviceToHostTimeMs = timer.Stop();
    timer.Start();

    // Free resources
    cudaFree(devVec);
    cudaFree(devVecResult);

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();

    measure.mReduceResult = sum;

    return measure;
}

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

ReduceFuncResult MeasureReductionGpuThrust(const std::vector<float>& hostVec, int threadsPerBlock)
{
    ReduceFuncResult measure("Thrust", 0, 0);
    CudaEventTimer totalTimer;
    totalTimer.Start();
    CudaEventTimer timer;
    timer.Start();

    thrust::device_vector<float> d_vec(hostVec.begin(), hostVec.end());

    measure.mCopyHostToDeviceTimeMs = timer.Stop();
    timer.Start();

    float sumOfElements = thrust::reduce(thrust::device, d_vec.begin(), d_vec.end());

    measure.mReduceResult = sumOfElements;
    measure.mReduceTimeMs = timer.Stop();
    timer.Start();

    d_vec.clear();
    d_vec.shrink_to_fit();

    measure.mFreeTimeMs = timer.Stop();
    measure.mTotalTime = totalTimer.Stop();
    return measure;
}

ReduceFuncResult  MeasureReductionCpu(const std::vector<float>& hostVec, int threadsPerBlock)
{
    ReduceFuncResult measure("Cpu", 0, 0);
    CTimerMicroseconds totalTimer;
    totalTimer.Start();
    CTimerMicroseconds timer;
    timer.Start();

#if 1
    float sumOfElements = std::accumulate(hostVec.begin(), hostVec.end(), 0.0);
#else
    double sumOfElements = 0;
    for (std::size_t i = 0; i < n; i++)
    {
        sumOfElements += hostVec[0];
    }
#endif

    measure.mReduceResult = sumOfElements;
    measure.mReduceTimeMs = timer.GetElapsed().count() * 0.001;

    measure.mTotalTime = totalTimer.GetElapsed().count() * 0.001;

    return measure;
}

ReduceFuncResult PrintToDisplay(ReduceFuncResult current)
{
    std::cout << std::endl << current.ToString();
    return std::move(current);
}

void SaveAndPrint(std::vector<ReduceFuncResult> results, ReduceFuncResult current)
{
    results.push_back(PrintToDisplay(std::move(current)));
    return;
}

void PrintProgressbar(const std::size_t i, const std::size_t n)
{
    const std::size_t cStep = n/20; // 5%
    static std::size_t currentTarget = 0;

    if (i == currentTarget + cStep)
    {
        std::cout << ".";
        currentTarget += cStep;
    }
    else
    if (i == n - 1)
    {
        std::cout << "." << " 100%";
        currentTarget = 0;
    }
}

int main()
{
    PrintCudeDeviceCaps();
    std::cout << std::endl << std::endl << std::endl;

    std::size_t n = 1000000000;

    std::cout << std::endl << "Generating " << n << " elements: ";
    // Initialize array on host
    std::vector<float> hostVec;
    hostVec.resize(n);
    for (std::size_t i = 0; i < n; i++)
    {
#if 1
        //hostVec[i] = 2.1f; // different result
        //hostVec[i] = 1.128f; // More different results
        //hostVec[i] = 1.1f; // almost same results
        //hostVec[i] = 1.0f; // All same results
        hostVec[i] = 0.75f; // All same results
        //hostVec[i] = 0.7523f; // All same results
#elif 0
        hostVec[i] = sinf(i*0.15) + cosf(i * 0.15);
#else
        float minValue = 0.1;
        float maxValue = 6.9;
        std::mt19937 prng(42);// same random seed for testing
        std::uniform_real_distribution<float> dist(minValue, maxValue);
        hostVec[i] = dist(prng);
#endif
        PrintProgressbar(i, n);
    }


    std::cout << std::endl << "First several elements: " << hostVec[0] << ", " << hostVec[1] << ", " << hostVec[2] << ", " << hostVec[3] << ", " << hostVec[4] << std::endl;
    std::cout << std::endl << "Reduce " << n << " elements";
    std::cout << std::endl << "Running kernel routines..." << std::endl;

    std::cout << std::endl << "Timings: " << std::endl;

    std::cout << std::endl << ReduceFuncResult::ToStringHeader();

    std::vector<ReduceFuncResult> results;
    SaveAndPrint(results, reduce1(hostVec, 32));
    SaveAndPrint(results, reduce1(hostVec, 64));
    SaveAndPrint(results, reduce1(hostVec, 128));
    SaveAndPrint(results, reduce1(hostVec, 256));
    SaveAndPrint(results, reduce1(hostVec, 512));
    SaveAndPrint(results, reduce1(hostVec, 1024));

    SaveAndPrint(results, reduce2(hostVec, 32));
    SaveAndPrint(results, reduce2(hostVec, 64));
    SaveAndPrint(results, reduce2(hostVec, 128));
    SaveAndPrint(results, reduce2(hostVec, 256));
    SaveAndPrint(results, reduce2(hostVec, 512));
    SaveAndPrint(results, reduce2(hostVec, 1024));

    SaveAndPrint(results, reduce3(hostVec, 32));
    SaveAndPrint(results, reduce3(hostVec, 64));
    SaveAndPrint(results, reduce3(hostVec, 128));
    SaveAndPrint(results, reduce3(hostVec, 256));
    SaveAndPrint(results, reduce3(hostVec, 512));
    SaveAndPrint(results, reduce3(hostVec, 1024));

    SaveAndPrint(results, reduce4(hostVec, 32));
    SaveAndPrint(results, reduce4(hostVec, 64));
    SaveAndPrint(results, reduce4(hostVec, 128));
    SaveAndPrint(results, reduce4(hostVec, 256));
    SaveAndPrint(results, reduce4(hostVec, 512));
    SaveAndPrint(results, reduce4(hostVec, 1024));

    SaveAndPrint(results, reduce6(hostVec, 32));
    SaveAndPrint(results, reduce6(hostVec, 64));
    SaveAndPrint(results, reduce6(hostVec, 128));
    SaveAndPrint(results, reduce6(hostVec, 256));
    SaveAndPrint(results, reduce6(hostVec, 512));
    SaveAndPrint(results, reduce6(hostVec, 1024));

    SaveAndPrint(results, reduce10(hostVec, 32));
    SaveAndPrint(results, reduce10(hostVec, 64));
    SaveAndPrint(results, reduce10(hostVec, 128));
    SaveAndPrint(results, reduce10(hostVec, 256));
    SaveAndPrint(results, reduce10(hostVec, 512));
    SaveAndPrint(results, reduce10(hostVec, 1024));

    SaveAndPrint(results, reduce20(hostVec, 32));
    SaveAndPrint(results, reduce20(hostVec, 64));
    SaveAndPrint(results, reduce20(hostVec, 128));
    SaveAndPrint(results, reduce20(hostVec, 256));
    SaveAndPrint(results, reduce20(hostVec, 512));
    SaveAndPrint(results, reduce20(hostVec, 1024));

    // last two: thrust and CPU
    SaveAndPrint(results, MeasureReductionGpuThrust(hostVec, 0));
    SaveAndPrint(results, MeasureReductionCpu(hostVec, 0));

    // Compare results but they're always different...
    // So see printed table
    //for (const ReduceFuncResult& result : results)
    //{
        //std::cout << std::endl << result.ToString();
    //}

    std::cout << std::endl << std::endl << "Program completed." << std::endl;
    return 0;
}

