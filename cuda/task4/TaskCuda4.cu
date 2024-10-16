
#include <algorithm>
#include <stdio.h>

//https://nvidia.github.io/cccl/thrust/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

#include "../common/MathTypesUtils.h"
#include "../common/OsUtilsCuda.h"

void MeasureThrustSort(std::size_t count)
{
    // Generate 32M random numbers serially.
    thrust::default_random_engine rng(1337);
    thrust::uniform_int_distribution<int> dist;
    thrust::host_vector<int> host_vec(count);
    thrust::generate(host_vec.begin(), host_vec.end(), [&] { return dist(rng); });

    // Transfer data to the device.
    thrust::device_vector<int> d_vec = host_vec;

    std::cout << std::endl << "Run thrust::sort... " << count << " elements";

    CudaEventTimer cudaTimer;
    cudaTimer.Start();
    CTimerMicroseconds chronoTimer;
    chronoTimer.Start();

    // Sort data on the device.
    thrust::sort(d_vec.begin(), d_vec.end());

    std::cout << std::endl << "CPU timer: " << chronoTimer.GetElapsed().count() << " microseconds";
    std::cout << std::endl << "GPU timer: " << cudaTimer.Stop() << " milliseconds";

    // Transfer data back to host.
    thrust::copy(d_vec.begin(), d_vec.end(), host_vec.begin());

    PrintFirstArrayElements(host_vec.data(), 15);
}

template<class T>
void insertionSort(T* array, std::size_t count)
{
    for (std::size_t i = 1; i < count; i++)
    {
        std::size_t j = i - 1;
        while (j >= 0 && array[j] > array[j + 1])
        {
            std::swap(array[j], array[j + 1]);
            //cout << "\ndid";
            j--;
        }
    }
}

void MeasureCpuStdSort(std::size_t count)
{
    // Generate 32M random numbers serially.
    thrust::default_random_engine rng(1337);
    thrust::uniform_int_distribution<int> dist;
    thrust::host_vector<int> host_vec(count);
    thrust::generate(host_vec.begin(), host_vec.end(), [&] { return dist(rng); });

    std::cout << std::endl << "Run std::sort... " << count << " elements";

    CTimerMicroseconds chronoTimer;
    chronoTimer.Start();

    // Sort data on the host.
    std::sort(host_vec.begin(), host_vec.end());

    std::cout << std::endl << "CPU timer: " << chronoTimer.GetElapsed().count() << " microseconds";

    PrintFirstArrayElements(host_vec.data(), 15);
}

void MeasureCpuInsertionSort(std::size_t count)
{
    // Generate 32M random numbers serially.
    thrust::default_random_engine rng(1337);
    thrust::uniform_int_distribution<int> dist;
    thrust::host_vector<int> host_vec(count);
    thrust::generate(host_vec.begin(), host_vec.end(), [&] { return dist(rng); });

    std::cout << std::endl << "Run insertion sort... " << count << " elements";

    CTimerMicroseconds chronoTimer;
    chronoTimer.Start();

    // Sort data on the host.
    insertionSort(host_vec.data(), host_vec.size());

    std::cout << std::endl << "CPU timer: " << chronoTimer.GetElapsed().count() << " microseconds";

    PrintFirstArrayElements(host_vec.data(), 15);
}

int main()
{
    std::size_t count = 1000000;
    MeasureThrustSort(count);
    MeasureCpuStdSort(count);
    MeasureCpuInsertionSort(count);
}