#ifndef CPURUN
#define CPURUN
#include "graphRead.hpp"
#include "param.cuh"
//#include "graphOperations.hpp"
#include "clusteringCount.hpp"
#include <nvToolsExt.h>

template <typename T>
void CPURun(Param param);
template <typename T>
void partitionedCPURun(Param param);
#endif
