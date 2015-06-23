template <typename T, unsigned int blockSize, unsigned int dataLength>
__device__ void conditionalWarpReduce(volatile T *sharedData)
{
  if(blockSize >= dataLength)
  {
    if(threadIdx.x < (dataLength/2))
    {sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
    __syncthreads();
  }
}

template <typename T, unsigned int blockSize>
__device__ void warpReduce(T* __restrict__ outDataPtr,
    volatile T* __restrict__ sharedData)
{
  conditionalWarpReduce<T, blockSize, 64>(sharedData);
  conditionalWarpReduce<T, blockSize, 32>(sharedData);
  conditionalWarpReduce<T, blockSize, 16>(sharedData);
  conditionalWarpReduce<T, blockSize, 8>(sharedData);
  conditionalWarpReduce<T, blockSize, 4>(sharedData);
  if(threadIdx.x == 0)
    {*outDataPtr= sharedData[0] + sharedData[1];}
  __syncthreads();
}

template <typename T, unsigned int blockSize, unsigned int dataLength>
__device__ void conditionalReduce(volatile T* __restrict__ sharedData)
{
  if(blockSize >= dataLength)
  {
    if(threadIdx.x < (dataLength/2))
    {sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
    __syncthreads();
  }

  if((blockSize < dataLength) && (blockSize > (dataLength/2)))
  {
    if(threadIdx.x+(dataLength/2) < blockSize)
    {sharedData[threadIdx.x] += sharedData[threadIdx.x+(dataLength/2)];}
    __syncthreads();
  }
}

template <typename T, unsigned int blockSize>
__device__ void blockReduce(T* __restrict__ outGlobalDataPtr,
    volatile T* __restrict__ sharedData)
{
  __syncthreads();
  conditionalReduce<T, blockSize, 1024>(sharedData);
  conditionalReduce<T, blockSize, 512>(sharedData);
  conditionalReduce<T, blockSize, 256>(sharedData);
  conditionalReduce<T, blockSize, 128>(sharedData);

  warpReduce<T, blockSize>(outGlobalDataPtr, sharedData);
  __syncthreads();
}
