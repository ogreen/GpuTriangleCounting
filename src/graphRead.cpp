#include "../include/graphRead.hpp"
template <typename T>
void readGraph(std::string filePath,
    T*& offset, T*& index, T& numVertices, T& numEdges)
{
  std::string infoFile = filePath;
  infoFile.append(".info");
  std::ifstream readGraphInfo(infoFile);
  readGraphInfo>>numVertices>>numEdges;
  offset = new T[numVertices];
  index = new T[numEdges];
  for(T i = 0; i < numVertices; i++)
  {
    T offsetInput;
    readGraphInfo>>offsetInput;
    offset[i] = offsetInput;
  }
  for(T i = 0; i < numEdges; i++)
  {
    T indexInput;
    readGraphInfo>>indexInput;
    index[i] = indexInput;
  }
  --numVertices;
  readGraphInfo.close();
}

template <typename T>
void readPartition(std::string filePath, T partitionCount,
    T*& partition, T& numVertices)
{
  std::string partitionFile = filePath;
  std::string ext = std::string(".part.") + std::to_string(partitionCount);
  partitionFile.append(ext);
  std::ifstream readGraphPartition(partitionFile);
  readGraphPartition>>numVertices;
  partition = new T[numVertices];
  for(T i = 0; i < numVertices; i++)
  {
    T vertex;
    readGraphPartition>>vertex;
    partition[i] = vertex;
  }
  readGraphPartition.close();
}

template void readGraph<int32_t>(std::string filePath,
    int32_t*& offset, int32_t*& index, int32_t& numVertices, int32_t& numEdges);

template void readGraph<int64_t>(std::string filePath,
    int64_t*& offset, int64_t*& index, int64_t& numVertices, int64_t& numEdges);

template void readPartition<int32_t>(std::string filePath,
    int32_t partitionCount, int32_t*& partition, int32_t& numVertices);

template void readPartition<int64_t>(std::string filePath,
    int64_t partitionCount, int64_t*& partition, int64_t& numVertices);
