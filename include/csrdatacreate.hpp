#ifndef CSRDATACREATE
#define CSRDATACREATE
#include <vector>

#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <omp.h>

template <typename T>
void readGraph(std::string fileIn,
    std::vector<T>& offset, std::vector<T>& index,
    T& numVertices, T& numEdges);

template <typename T>
void readGraph(std::string fileIn,
    T*& offset, T*& index,
    T& numVertices, T& numEdges);

template <typename T>
void writeGraph(std::string fileOut,
    const std::vector<T>& offset, const std::vector<T>& index);
#endif
