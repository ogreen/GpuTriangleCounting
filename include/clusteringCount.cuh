#ifndef CLUSTERINGGPU
#define CLUSTERINGGPU
#define MAX_THREADS     1024
#define CUDA(call, ...) do {                        \
        cudaError_t _e = (call);                    \
        if (_e == cudaSuccess) break;               \
        fprintf(stdout,                             \
                "CUDA runtime error: %s (%d)\n",    \
                cudaGetErrorString(_e), _e);        \
        return -1;                                  \
    } while (0)
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <chrono>
#include <ctime>
#include "reduce.cuh"
//extern std::vector<cudaStream_t> str;
template <typename T>
__device__ void initialize(const T diag_id,
    const T u_len, T v_len,
    T* const __restrict__ u_min, T* const __restrict__ u_max,
    T* const __restrict__ v_min, T* const __restrict__ v_max,
    T* const __restrict__ found);

template <typename T>
__device__ void calcWorkPerThread(const T uLength,
    const T vLength, const T threadsPerIntersection,
    const T threadId,
    T * const __restrict__ outWorkPerThread,
    T * const __restrict__ outDiagonalId);

template <typename T>
__device__ void bSearch(
    unsigned int found,
    const T diagonalId,
    T const * const __restrict__ uNodes,
    T const * const __restrict__ vNodes,
    T const * const __restrict__ uLength,
    T * const __restrict__ outUMin,
    T * const __restrict__ outUMax,
    T * const __restrict__ outVMin,
    T * const __restrict__ outVMax,
    T * const __restrict__ outUCurr,
    T * const __restrict__ outVCurr);

template <typename T>
__device__ T fixThreadWorkEdges(const T uLength, const T vLength,
    T * const __restrict__ uCurr, T * const __restrict__ vCurr,
    T const * const __restrict__ uNodes, T const * const __restrict__ vNodes);

template <typename T>
__device__ void intersectCount(const T uLength, const T vLength,
    T const * const __restrict__ uNodes, T const * const __restrict__ vNodes,
    T * const __restrict__ uCurr, T * const __restrict__ vCurr,
    T * const __restrict__ workIndex, T * const __restrict__ workPerThread,
    T * const __restrict__ triangles, T found, T lim);

template <typename T>
__device__ void intersectCount(const T uLength, const T vLength,
    T const * const __restrict__ uNodes, T const * const __restrict__ vNodes,
    T * const __restrict__ uCurr, T * const __restrict__ vCurr,
    T * const __restrict__ workIndex, T * const __restrict__ workPerThread,
    T * const __restrict__ triangles, T found);

template <typename T>
__device__ T count_triangles(T u, T const * const __restrict__ u_nodes, T u_len,
    T v, T const * const __restrict__ v_nodes, T v_len, T threads_per_block,
    volatile T* __restrict__ firstFound, T tId);

template <typename T>
__device__ T count_triangles(T u, T const * const __restrict__ u_nodes, T u_len,
    T v, T const * const __restrict__ v_nodes, T v_len, T threads_per_block,
    volatile T* __restrict__ firstFound, T tId, T lim);

template <typename T>
__device__ void calcWorkPerBlock(const T numVertices,
    T * const __restrict__ outMpStart,
    T * const __restrict__ outMpEnd);

template <typename T, unsigned int blockSize>
__global__ void count_all_trianglesGPU (const T nv,
    T const * const __restrict__ vertex, T const * const __restrict__ d_off,
    T const * const __restrict__ d_ind, T * const __restrict__ outPutTriangles,
    const T threads_per_block, const T number_blocks, const T shifter);

template <typename T, unsigned int blockSize>
__global__ void count_all_trianglesGPU (const T nv,
    T const * const __restrict__ d_off, T const * const __restrict__ d_ind,
    T * const __restrict__ outPutTriangles, const T threads_per_block,
    const T number_blocks, const T shifter);

template <typename T>
void kernelCall(unsigned int numberBlocks, unsigned int numberThreads,
    const T nv, T const * const __restrict__ vertex,
    T const * const __restrict__ d_off, T const * const __restrict__ d_ind,
    T * const __restrict__ outPutTriangles, const T threads_per_block,
    const T number_blocks, const T shifter);

template <typename T>
void kernelCall(unsigned int numberBlocks, unsigned int numberThreads,
    const T nv, T const * const __restrict__ d_off,
    T const * const __restrict__ d_ind,
    T * const __restrict__ outPutTriangles, const T threads_per_block,
    const T number_blocks, const T shifter);

template <typename T>
float timedKernelCall(unsigned int numberBlocks, unsigned int numberThreads,
    const T nv, T const * const __restrict__ vertex,
    T const * const __restrict__ d_off, T const * const __restrict__ d_ind,
    T * const __restrict__ outPutTriangles, const T threads_per_block,
    const T number_blocks, const T shifter);

template <typename T>
float timedKernelCall(unsigned int numberBlocks, unsigned int numberThreads,
    const T nv, T const * const __restrict__ d_off,
    T const * const __restrict__ d_ind,
    T * const __restrict__ outPutTriangles, const T threads_per_block,
    const T number_blocks, const T shifter);

#endif
