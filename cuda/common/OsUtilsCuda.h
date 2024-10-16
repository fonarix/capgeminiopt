

#ifndef __OsUtilsCuda_h__
#define __OsUtilsCuda_h__


#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

void PrintCudeDeviceCaps();
int GetCudeDefaultDevice();

class CCudaEvent
{
    cudaEvent_t mEvent;
    //cudaStream_t stream;
    cudaError_t mErr;
public:

    CCudaEvent(const CCudaEvent&) = delete;
    CCudaEvent& operator=(const CCudaEvent&) = delete;
    CCudaEvent(CCudaEvent&&) = delete;
    CCudaEvent& operator=(CCudaEvent&&) = delete;

    CCudaEvent();
    ~CCudaEvent();
    cudaEvent_t GetEvent();
    void Create();
    void Destroy();
    void Record();
    void Synchronize();
};

/*
 Common schema:

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // cuda calls

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Running cuda time: " << milliseconds  << " milliseconds clear cuda events" << std::endl;
 */
class CudaEventTimer
{
    // some events to count the execution time
    CCudaEvent  mStartEvent;
    CCudaEvent  mStopEvent;
    float       elapsed_time_ms = 0.0f;
    cudaError_t mErr;
    void        FreeEvents();

public:
    CudaEventTimer();
    ~CudaEventTimer();

    void Start();
    float Stop();
};

template <class T>
class TCudaDeviceArray
{
public:
    explicit TCudaDeviceArray()
        : start_(0)
        , size_(0)
    {}

    explicit TCudaDeviceArray(size_t size)
    {
        allocate(size);
    }

    ~TCudaDeviceArray()
    {
        free();
    }

    // resize the vector
    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    // get the size of the array
    size_t getSize() const
    {
        return size_;
    }

    // get data
    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    // set
    void set(const T* src, size_t count)
    {
        resize(count);
        cudaError_t result = cudaMemcpy(start_, src, count * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    // get
    void get(T* dest, size_t count)
    {
        cudaError_t result = cudaMemcpy(dest, start_, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }

    // private functions
private:
    // allocate memory on the device
    void allocate(size_t count)
    {
        cudaError_t result = cudaMalloc((void**)&start_, count * sizeof(T));
        if (result != cudaSuccess)
        {
            start_ = 0;
            throw std::runtime_error("failed to allocate device memory");
        }
        size_ = count;
    }

    // free memory on the device
    void free()
    {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = 0;
        }
    }

    T* start_;
    size_t size_;
};

using CudaDeviceArrayFloat = TCudaDeviceArray<float>;


#endif


