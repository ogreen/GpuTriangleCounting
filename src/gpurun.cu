#include "../include/gpurun.cuh"
template <typename T>
void singleParamTestGPURun(Param param)
{
  T* offsetVector;
  T* indexVector;
  T vertexCount;
  T edgeCount;

  {
    thrust::device_vector<int> memory(1);
  }

  readGraph(param.fileName, offsetVector, indexVector,
      vertexCount, edgeCount);

  std::chrono::time_point<std::chrono::system_clock> execStart, execEnd,
    memAllocEnd;
  execStart = std::chrono::system_clock::now();
  thrust::device_vector<T> dOffsetVector(offsetVector,
      offsetVector + vertexCount + 1);
  thrust::device_vector<T> dIndexVector(indexVector, indexVector + edgeCount);
  thrust::device_vector<T> dTriangleOutputVector(vertexCount, 0);

  T const * const dOffset = thrust::raw_pointer_cast(dOffsetVector.data());
  T const * const dIndex = thrust::raw_pointer_cast(dIndexVector.data());
  T * const dTriangle = thrust::raw_pointer_cast(dTriangleOutputVector.data());
  cudaDeviceSynchronize();
  memAllocEnd = std::chrono::system_clock::now();

  unsigned int blocks = param.blocks;
  unsigned int blockSize = param.threadCount;
  T threadsPerIntsctn = param.threadPerInt;
  T intsctnPerBlock = param.threadCount/param.threadPerInt;
  T threadShift = std::log2(param.threadPerInt);
  kernelCall(blocks, blockSize, vertexCount, dOffset,
      dIndex, dTriangle, threadsPerIntsctn, intsctnPerBlock, threadShift);
  cudaDeviceSynchronize();
  execEnd = std::chrono::system_clock::now();
  T totalTriangleCount = thrust::reduce(dTriangleOutputVector.begin(),
      dTriangleOutputVector.end());
  std::chrono::duration<float, std::milli> memAllocDuration = memAllocEnd -
    execStart;
  std::chrono::duration<float, std::milli> tCountDuration = execEnd -
    memAllocEnd;
  std::chrono::duration<float, std::milli> execDuration = execEnd -
    execStart;
  
  std::cout<<vertexCount<<"\t"<<totalTriangleCount<<"\t"<<
    memAllocDuration.count()<<"\t"<<tCountDuration.count()<<"\t"<<
    execDuration.count()<<"\n";
    

  delete[] offsetVector;
  delete[] indexVector;
}


template <typename T>
void singleGPURun(Param param,
    T* offsetVector, T vertexCount, T* indexVector, T edgeCount)
{
  {
    thrust::device_vector<int> memory(1);
  }
  std::string fileName = std::string("runresult/") + param.fileName +
    std::string(".o.") + std::to_string(param.blocks) + std::string(".") +
    std::to_string(param.threadCount) + std::string(".") +
    std::to_string(param.threadPerInt);
  std::ofstream fout(fileName, std::ios::out|std::ios::app);

  std::chrono::time_point<std::chrono::system_clock> execStart, execEnd,
    memAllocEnd;
  execStart = std::chrono::system_clock::now();
  thrust::device_vector<T> dOffsetVector(offsetVector,
      offsetVector + vertexCount + 1);
  thrust::device_vector<T> dIndexVector(indexVector, indexVector + edgeCount);
  thrust::device_vector<T> dTriangleOutputVector(vertexCount, 0);

  T const * const dOffset = thrust::raw_pointer_cast(dOffsetVector.data());
  T const * const dIndex = thrust::raw_pointer_cast(dIndexVector.data());
  T * const dTriangle = thrust::raw_pointer_cast(dTriangleOutputVector.data());
  cudaDeviceSynchronize();
  memAllocEnd = std::chrono::system_clock::now();

  unsigned int blocks = param.blocks;
  unsigned int blockSize = param.threadCount;
  T threadsPerIntsctn = param.threadPerInt;
  T intsctnPerBlock = param.threadCount/param.threadPerInt;
  T threadShift = std::log2(param.threadPerInt);
  kernelCall(blocks, blockSize, vertexCount, dOffset,
      dIndex, dTriangle, threadsPerIntsctn, intsctnPerBlock, threadShift);
  cudaDeviceSynchronize();
  execEnd = std::chrono::system_clock::now();
  T totalTriangleCount = thrust::reduce(dTriangleOutputVector.begin(),
      dTriangleOutputVector.end());
  std::chrono::duration<float, std::milli> memAllocDuration = memAllocEnd -
    execStart;
  std::chrono::duration<float, std::milli> tCountDuration = execEnd -
    memAllocEnd;
  std::chrono::duration<float, std::milli> execDuration = execEnd -
    execStart;
  fout<<"ctime\t1\t"<<tCountDuration.count()<<"\n\n";
  fout.close();
  /*
  std::cout<<vertexCount<<"\t"<<totalTriangleCount<<"\t"<<
    memAllocDuration.count()<<"\t"<<tCountDuration.count()<<"\t"<<
    execDuration.count()<<"\n";
    */

}



template <typename T>
void allParamTestGPURun(Param param)
{
  T* offsetVector;
  T* indexVector;
  T vertexCount;
  T edgeCount;

  {
    thrust::device_vector<int> memory(1);
  }

  readGraph(param.fileName, offsetVector, indexVector,
      vertexCount, edgeCount);
  cudaDeviceSynchronize();

  std::chrono::time_point<std::chrono::system_clock> memAllocStart,
    memAllocEnd;
  memAllocStart = std::chrono::system_clock::now();
  thrust::device_vector<T> dOffsetVector(offsetVector,
      offsetVector + vertexCount + 1);
  thrust::device_vector<T> dIndexVector(indexVector, indexVector + edgeCount);
  thrust::device_vector<T> dTriangleOutputVector(dOffsetVector.size(), 0);

  T const * const dOffset = thrust::raw_pointer_cast(dOffsetVector.data());
  T const * const dIndex = thrust::raw_pointer_cast(dIndexVector.data());
  T * const dTriangle = thrust::raw_pointer_cast(dTriangleOutputVector.data());
  memAllocEnd = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> memAllocDuration = memAllocEnd -
    memAllocStart;

  std::string separator = std::string(".o.");
  std::string fileOutName = param.fileName + separator +
    std::to_string(param.blocks);
  std::ofstream writeFile(fileOutName);

  for(auto paramBlockSize : globalParam::blockSizeParam)
  {
    for(auto paramThreadsPerIntsctn : globalParam::threadPerIntersectionParam)
    {
      thrust::fill(dTriangleOutputVector.begin(), dTriangleOutputVector.end(),
          0);
      std::chrono::time_point<std::chrono::system_clock> execStart, execEnd;
      execStart = std::chrono::system_clock::now();
      unsigned int blocks = param.blocks;
      unsigned int blockSize = paramBlockSize;
      T threadsPerIntsctn = paramThreadsPerIntsctn;
      T intsctnPerBlock = paramBlockSize/paramThreadsPerIntsctn;
      T threadShift = std::log2(paramThreadsPerIntsctn);
      kernelCall(blocks, blockSize, vertexCount, dOffset,
          dIndex, dTriangle, threadsPerIntsctn, intsctnPerBlock, threadShift);
      T sumTriangles = thrust::reduce(dTriangleOutputVector.begin(), dTriangleOutputVector.end());
      execEnd = std::chrono::system_clock::now();
      std::chrono::duration<float, std::milli> execDuration = execEnd -
        execStart;
      writeFile<<paramBlockSize<<"\t"<<paramThreadsPerIntsctn<<"\t"<<
        memAllocDuration.count()<<"\t"<<execDuration.count()<<"\t"<<
        execDuration.count()+memAllocDuration.count()<<"\t"<<sumTriangles<<"\n";
    }
  }
  writeFile.close();
}


template void singleParamTestGPURun<int32_t>(Param param);

template void singleGPURun<int32_t>(Param param,
    int32_t* offsetVector, int32_t vertexCount,
    int32_t* indexVector, int32_t edgeCount);
template void allParamTestGPURun<int32_t>(Param param);

template void singleParamTestGPURun<int64_t>(Param param);
template void singleGPURun<int64_t>(Param param,
    int64_t* offsetVector, int64_t vertexCount,
    int64_t* indexVector, int64_t edgeCount);

template void allParamTestGPURun<int64_t>(Param param);
