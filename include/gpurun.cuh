#ifndef GPURUN
#define GPURUN
#include "graphRead.hpp"
#include "param.cuh"
//#include "graphOperations.hpp"
#include "clusteringCount.cuh"
#include <nvToolsExt.h>
#include <omp.h>
template <typename T>
void mapPartition(std::vector<unsigned int> dMap, unsigned int numDevices,
    T* partitionDeviceMap, T pCount);

template <typename T>
void singleParamTestGPURun(Param param);

template <typename T>
void partitionedSingleParamGPURun(Param param);

template <typename T>
void parallelPartitionedSingleParamGPURun(Param param);

template <typename T>
void multiGPUParallelPartitionedSingleParamGPURun(Param param);

template <typename T>
void singleGPURun(Param param,
    T* offsetVector, T vertexCount, T* indexVector, T edgeCount);

template <typename T>
void multiGPUParallelPartitionedMultiParamGPURun(Param param,
    T* offsetVector, T vertexCount, T* indexVector, T edgeCount);

template <typename T>
void allParamTestGPURun(Param param);
#endif
