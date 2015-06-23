#ifndef GRAPHREAD
#define GRAPHREAD
#include<string>
#include<fstream>
template <typename T>
void readGraph(std::string filePath,
    T*& offset, T*& index, T& numVertices, T& numEdges);

template <typename T>
void readPartition(std::string filePath, T partitionCount,
    T*& partition, T& numVertices);
#endif
