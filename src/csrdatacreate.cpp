#include "../include/csrdatacreate.hpp"

template <typename T>
void readGraph(std::string fileIn,
    std::vector<T>& offset, std::vector<T>& index,
    T& numVertices, T& numEdges)
{
  std::ifstream readFile (fileIn.c_str());
  std::string graphInfoString;
  std::getline(readFile, graphInfoString);
  while(graphInfoString.at(0) == '%')
  {
    std::getline(readFile, graphInfoString);
  }
  std::istringstream graphInfo(graphInfoString);

  graphInfo>>numVertices>>numEdges;

  numEdges *= 2;
  offset.reserve(numVertices+1);
  index.reserve(numEdges);
  offset.shrink_to_fit();
  index.shrink_to_fit();

  offset.push_back(0);

  std::string adjacencyListString;
  T lineCount = 0;
  while((std::getline(readFile, adjacencyListString)) &&
      (lineCount < numVertices))
  {
    lineCount++;
    if(adjacencyListString.size() == 0)
    {
      offset.push_back(index.size());
      continue;
    }
    std::istringstream adjacencyStream(adjacencyListString);
    T vertex;
    auto startIndex = index.size();
    while(adjacencyStream>>vertex)
    {
      index.push_back(vertex-1);
    }
    std::sort(index.begin() + startIndex, index.end());
    offset.push_back(index.size());
  }
  readFile.close();
}

template <typename T>
void readGraph(std::string fileIn,
    T*& offset, T*& index, T& numVertices, T& numEdges)
{
  std::ifstream readFile (fileIn.c_str());
  std::string graphInfoString;
  std::getline(readFile, graphInfoString);
  std::istringstream graphInfo(graphInfoString);

  graphInfo>>numVertices>>numEdges;

  numEdges *= 2;
  offset = new T[numVertices+1];
  index = new T[numEdges];

  T offsetCount = 0;
  T indexCount = 0;
  offset[offsetCount++] = 0;
  std::string adjacencyListString;
  while(std::getline(readFile, adjacencyListString))
  {
    std::istringstream adjacencyStream(adjacencyListString);
    T vertex;
    T startIndex = indexCount;
    while(adjacencyStream>>vertex)
    {
      index[indexCount++] = vertex-1;
    }
    T endIndex = indexCount;
    std::sort(&index[startIndex], &index[endIndex]);
    offset[offsetCount++] = endIndex;
  }
  readFile.close();
}

template <typename T>
void writeGraph(std::string fileOut,
    const std::vector<T>& offset, const std::vector<T>& index)
{
  std::ofstream writeFile (fileOut.c_str());
  writeFile<<offset.size()<<"\t"<<index.size()<<"\n";
  bool addSeparator = false;
  for(const auto& o : offset)
  {
    if(addSeparator)
    {
      writeFile<<"\t";
    }
    writeFile<<o;
    addSeparator = true;
  }
  writeFile<<"\n";
  addSeparator = false;
  for(const auto& i : index)
  {
    if(addSeparator)
    {
      writeFile<<"\t";
    }
    writeFile<<i;
    addSeparator = true;
  }
  writeFile.close();
}

