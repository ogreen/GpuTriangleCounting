#ifndef CLUSTERINGCPU
#define CLUSTERINGCPU
#include <vector>
#include <chrono>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
int64_t count_intersectionsReg (const T ai, const T alen,
    const T * a, const T bi, const T blen, const T * b);

template <typename T>
void count_all_trianglesCPU (const T nv, const T * off,
    const T * ind, int64_t * triNE,
    int64_t* allTriangles);

template <typename T>
void count_all_trianglesCPU (const T nv, const std::vector<T> off,
    const std::vector<T> ind, int64_t * triNE,
    int64_t* allTriangles);

template <typename T>
void count_all_trianglesCPU (const T nv, const std::vector<T> vertexList,
    const std::vector<T> off, const std::vector<T> ind, int64_t * triNE,
    int64_t* allTriangles);

template <typename T>
void count_all_trianglesCPU (const T nv, const T* vertexList,
    const T* off, const T* ind, int64_t * triNE,
    int64_t* allTriangles);

template <typename T>
int64_t cpuRun(T nv, T ne, T * off, T * ind, float& timeCPU);

template <typename T>
int64_t cpuRun(T nv, T ne, std::vector<T> off, std::vector<T> ind,
    float& timeCPU);

template <typename T>
int64_t cpuRun(T nv, T ne, std::vector<T> vertexList, std::vector<T> off,
    std::vector<T> ind, float& timeCPU);

template <typename T>
int64_t cpuRun(T nv, T ne, T* vertexList, T* off,
    T* ind, float& timeCPU);
#endif
