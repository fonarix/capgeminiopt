


#include "OsUtilsCuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <sstream>

/* Links
 * https://www.youtube.com/watch?v=TiE_KMbI3Wo&list=PLdhEOd5Bckb6eDz0S3gQ27mxCKT2WFHuU&index=3
 *
 *
 *
 *
 *
 * kernelName<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
 */


namespace {

template<typename T = int, char sep = ' ', int size>
std::string arrayToStr(T(&inArray)[size])
{
    std::stringstream stream;
    stream << std::dec;
    if (size > 0)
    {
        stream << std::dec << inArray[0];
    }

    for (auto i = 1; i < size; i++)
    {
        stream << std::dec << sep << inArray[i];
    }

    return stream.str();
}

}

std::string GetDevicePropToString(cudaDeviceProp inDevProp)
{
    std::stringstream ss;

    ss << "\nDevice name....: " << inDevProp.name;
    ss << "\nGlobal memory available on device in bytes.........: " << inDevProp.totalGlobalMem;
    ss << "\nShared memory available per block in bytes.........: " << inDevProp.sharedMemPerBlock;
    ss << "\nCount of 32-bit registers available per block......: " << inDevProp.regsPerBlock;
    ss << "\n32-bit registers available per multiprocessor......: " << inDevProp.regsPerMultiprocessor;
    ss << "\nWarp size in threads...............................: " << inDevProp.warpSize;
    ss << "\nMaximum pitch in bytes allowed by memory copies....: " << inDevProp.memPitch;
    ss << "\nMaximum number of threads per block................: " << inDevProp.maxThreadsPerBlock;
    ss << "\nMaximum size of each dimension of a block..........: " << arrayToStr(inDevProp.maxThreadsDim);
    ss << "\nMaximum size of each dimension of a grid...........: " << arrayToStr(inDevProp.maxGridSize);
    ss << "\nDeprecated, Clock frequency in kilohertz...........: " << inDevProp.clockRate;
    ss << "\nConstant memory available on device in bytes.......: " << inDevProp.totalConstMem;
    ss << "\nMajor compute capability...............: " << inDevProp.major;
    ss << "\nMinor compute capability...............: " << inDevProp.minor;
    ss << "\nAlignment requirement for textures.....: " << inDevProp.textureAlignment;
    ss << "\nPitch alignment requirement for texture references bound to pitched memory: " << inDevProp.texturePitchAlignment;
    ss << "\nDevice can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.: " << inDevProp.deviceOverlap;
    ss << "\nNumber of multiprocessors on device....: " << inDevProp.multiProcessorCount;
    ss << "\nDeprecated, Specified whether there is a run time limit on kernels: " << inDevProp.kernelExecTimeoutEnabled;
    ss << "\nDevice is integrated as opposed to discrete: " << inDevProp.integrated;
    ss << "\nDevice can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: " << inDevProp.canMapHostMemory;
    ss << "\nDeprecated, Compute mode (See ::cudaComputeMode): " << inDevProp.computeMode;
    ss << "\nMaximum 1D texture size................: " << inDevProp.maxTexture1D;
    ss << "\nMaximum 1D mipmapped texture size......: " << inDevProp.maxTexture1DMipmap;
    ss << "\nDeprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.: " << inDevProp.maxTexture1DLinear;
    ss << "\nMaximum 2D texture dimensions..........: " << arrayToStr(inDevProp.maxTexture2D);
    ss << "\nMaximum 2D mipmapped texture dimensions: " << arrayToStr(inDevProp.maxTexture2DMipmap);
    ss << "\nMaximum dimensions (width, height, pitch) for 2D textures bound to pitched memory..: " << arrayToStr(inDevProp.maxTexture2DLinear);
    ss << "\nMaximum 2D texture dimensions if texture gather operations have to be performed....: " << arrayToStr(inDevProp.maxTexture2DGather);
    ss << "\nMaximum 3D texture dimensions..........: " << arrayToStr(inDevProp.maxTexture3D);
    ss << "\nMaximum alternate 3D texture dimensions: " << arrayToStr(inDevProp.maxTexture3DAlt);
    ss << "\nMaximum Cubemap texture dimensions.....: " << inDevProp.maxTextureCubemap;
    ss << "\nMaximum 1D layered texture dimensions..: " << arrayToStr(inDevProp.maxTexture1DLayered);
    ss << "\nMaximum 2D layered texture dimensions..: " << arrayToStr(inDevProp.maxTexture2DLayered);
    ss << "\nMaximum Cubemap layered texture dimensions: " << arrayToStr(inDevProp.maxTextureCubemapLayered);
    ss << "\nMaximum 1D surface size................: " << inDevProp.maxSurface1D;
    ss << "\nMaximum 2D surface dimensions..........: " << arrayToStr(inDevProp.maxSurface2D);
    ss << "\nMaximum 3D surface dimensions..........: " << arrayToStr(inDevProp.maxSurface3D);
    ss << "\nMaximum 1D layered surface dimensions..: " << arrayToStr(inDevProp.maxSurface1DLayered);
    ss << "\nMaximum 2D layered surface dimensions..: " << arrayToStr(inDevProp.maxSurface2DLayered);
    ss << "\nMaximum Cubemap surface dimensions.....: " << inDevProp.maxSurfaceCubemap;
    ss << "\nMaximum Cubemap layered surface dimensions: " << arrayToStr(inDevProp.maxSurfaceCubemapLayered);


    ss << "\nAlignment requirements for surfaces....: " << inDevProp.surfaceAlignment;
    ss << "\nDevice can possibly execute multiple kernels concurrently: " << inDevProp.concurrentKernels;
    ss << "\nDevice has ECC support enabled.........: " << inDevProp.ECCEnabled;
    ss << "\nPCI bus ID of the device...............: " << inDevProp.pciBusID;
    ss << "\nPCI device ID of the device............: " << inDevProp.pciDeviceID;
    ss << "\nPCI domain ID of the device............: " << inDevProp.pciDomainID;
    ss << "\n1 if device is a Tesla device using TCC driver, 0 otherwise: " << inDevProp.tccDriver;
    ss << "\nNumber of asynchronous engines.........: " << inDevProp.asyncEngineCount;
    ss << "\nDevice shares a unified address space with the host....: " << inDevProp.unifiedAddressing;
    ss << "\nDeprecated, Peak memory clock frequency in kilohertz...: " << inDevProp.memoryClockRate;
    ss << "\nGlobal memory bus width in bits........: " << inDevProp.memoryBusWidth;
    ss << "\nSize of L2 cache in bytes..............: " << inDevProp.l2CacheSize;
    ss << "\nDevice's maximum l2 persisting lines capacity setting in bytes: " << inDevProp.persistingL2CacheMaxSize;
    ss << "\nMaximum resident threads per multiprocessor: " << inDevProp.maxThreadsPerMultiProcessor;
    ss << "\nDevice supports stream priorities......: " << inDevProp.streamPrioritiesSupported;
    ss << "\nDevice supports caching globals in L1..: " << inDevProp.globalL1CacheSupported;
    ss << "\nDevice supports caching locals in L1...: " << inDevProp.localL1CacheSupported;
    ss << "\nShared memory available per multiprocessor in bytes........: " << inDevProp.sharedMemPerMultiprocessor;
    ss << "\n32-bit registers available per multiprocessor: " << inDevProp.regsPerMultiprocessor;
    ss << "\nDevice supports allocating managed memory on this system...: " << inDevProp.managedMemory;
    ss << "\nDevice is on a multi-GPU board.........: " << inDevProp.isMultiGpuBoard;
    ss << "\nUnique identifier for a group of devices on the same multi-GPU board: " << inDevProp.multiGpuBoardGroupID;
    ss << "\nLink between the device and the host supports native atomic operations: " << inDevProp.hostNativeAtomicSupported;
    ss << "\nDeprecated, Ratio of single precision performance (in floating-point operations per second) to double precision performance: " << inDevProp.singleToDoublePrecisionPerfRatio;
    ss << "\nDevice supports coherently accessing pageable memory without calling cudaHostRegister on it: " << inDevProp.pageableMemoryAccess;
    ss << "\nDevice can coherently access managed memory concurrently with the CPU: " << inDevProp.concurrentManagedAccess;
    ss << "\nDevice supports Compute Preemption.....: " << inDevProp.computePreemptionSupported;
    ss << "\nDevice can access host registered memory at the same virtual address as the CPU: " << inDevProp.canUseHostPointerForRegisteredMem;
    ss << "\nDevice supports launching cooperative kernels via ::cudaLaunchCooperativeKernel: " << inDevProp.cooperativeLaunch;
    ss << "\nDeprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated...: " << inDevProp.cooperativeMultiDeviceLaunch;
    ss << "\nPer device maximum shared memory per block usable by special opt in: " << inDevProp.sharedMemPerBlockOptin;
    ss << "\nDevice accesses pageable memory via the host's page tables: " << inDevProp.pageableMemoryAccessUsesHostPageTables;
    ss << "\nHost can directly access managed memory on the device without migration.: " << inDevProp.directManagedMemAccessFromHost;
    ss << "\nMaximum number of resident blocks per multiprocessor.......: " << inDevProp.maxBlocksPerMultiProcessor;
    ss << "\nThe maximum value of ::cudaAccessPolicyWindow::num_bytes...: " << inDevProp.accessPolicyMaxWindowSize;
    ss << "\nShared memory reserved by CUDA driver per block in bytes...: " << inDevProp.reservedSharedMemPerBlock;
    ss << "\nDevice supports host memory registration via ::cudaHostRegister: " << inDevProp.hostRegisterSupported;
    ss << "\n1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise: " << inDevProp.sparseCudaArraySupported;
    ss << "\nDevice supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU: " << inDevProp.hostRegisterReadOnlySupported;
    ss << "\nExternal timeline semaphore interop is supported on the device.: " << inDevProp.timelineSemaphoreInteropSupported;
    ss << "\n1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise: " << inDevProp.memoryPoolsSupported;
    ss << "\n1 if the device supports GPUDirect RDMA APIs, 0 otherwise..: " << inDevProp.gpuDirectRDMASupported;
    ss << "\nBitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum: " << inDevProp.gpuDirectRDMAFlushWritesOptions;
    ss << "\nSee the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values: " << inDevProp.gpuDirectRDMAWritesOrdering;
    ss << "\nBitmask of handle types supported with mempool-based IPC...: " << inDevProp.memoryPoolSupportedHandleTypes;
    ss << "\n1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays: " << inDevProp.deferredMappingCudaArraySupported;
    ss << "\nDevice supports IPC Events.................: " << inDevProp.ipcEventSupported;
    ss << "\nIndicates device supports cluster launch...: " << inDevProp.clusterLaunch;
    ss << "\nIndicates device supports unified pointers.: " << inDevProp.unifiedFunctionPointers;

    return ss.str();
}

int GetSPcores(cudaDeviceProp inDevProp)
{
    std::stringstream ss;
    int cores = 0;
    int mp = inDevProp.multiProcessorCount;
    inDevProp.major;
    return 0;
}

void PrintCudeDeviceCaps()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (cudaSuccess != error_id)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        return;
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::string info = GetDevicePropToString(deviceProp);
        std::cout << info;
    }
}

int GetCudeDefaultDevice()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        return -1;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        return -1;
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0;
    /*
    cudaDeviceProp devProp;
    memset(&devProp, 0x00, sizeof(cudaDeviceProp));
    devProp.major = 1;
    devProp.minor = 3;
    cudaChooseDevice(&dev , &devProp);
    cudaSetDevice(dev);
    //*/

    return dev;
}

CCudaEvent::CCudaEvent()
    : mEvent(nullptr)
{
}

CCudaEvent::~CCudaEvent()
{
    Destroy();
}

cudaEvent_t CCudaEvent::GetEvent()
{
    return mEvent;
};

void CCudaEvent::Create()
{
    Destroy();
    mErr = cudaEventCreate(&mEvent);
}

void CCudaEvent::Destroy()
{
    if (mEvent)
    {
        mErr = cudaEventDestroy(mEvent);
        mEvent = nullptr;
    }
}

void CCudaEvent::Record()
{
    mErr = cudaEventRecord(mEvent, 0);
}

void CCudaEvent::Synchronize()
{
    mErr = cudaEventSynchronize(mEvent);
}

//
void CudaEventTimer::FreeEvents()
{
    mStartEvent.Destroy();
    mStopEvent.Destroy();
}

CudaEventTimer::CudaEventTimer()
 : mErr(cudaErrorInvalidValue)
{
}

CudaEventTimer::~CudaEventTimer()
{
    FreeEvents();
}

void CudaEventTimer::Start()
{
    mStartEvent.Create();
    mStopEvent.Create();
    // start to count execution time of GPU version
    mStartEvent.Record();
}

float CudaEventTimer::Stop()
{
    // start to count execution time of GPU version
    mStopEvent.Record();
    mStopEvent.Synchronize();
    // compute time elapse on GPU or CPU computing
    elapsed_time_ms = 0.0f;
    mErr = cudaEventElapsedTime(&elapsed_time_ms, mStartEvent.GetEvent(), mStopEvent.GetEvent());
    return elapsed_time_ms;
}
