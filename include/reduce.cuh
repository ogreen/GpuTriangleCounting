#ifndef REDUCE
#define REDUCE
template <typename T, unsigned int blockSize, unsigned int dataLength>
__device__ void conditionalWarpReduce(volatile T *sharedData);

template <typename T, unsigned int blockSize>
__device__ void warpReduce(T* __restrict__ outDataPtr,
    volatile T* __restrict__ sharedData);

template <typename T, unsigned int blockSize, unsigned int dataLength>
__device__ void conditionalReduce(volatile T* __restrict__ sharedData);

template <typename T, unsigned int blockSize>
__device__ void blockReduce(T* __restrict__ outGlobalDataPtr,
    volatile T* __restrict__ sharedData);

#include "../src/reduce.cu"
#endif
