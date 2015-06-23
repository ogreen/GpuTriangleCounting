#include "../include/clusteringCount.hpp"
template <typename T>
int64_t count_intersectionsReg (const T ai, const T alen,
    const T * a, const T bi, const T blen, const T * b)
{
  T ka = 0, kb = 0;
  int64_t out = 0;

  if (!alen || !blen || a[alen-1] < b[0] || b[blen-1] < a[0])
    return 0;

  while (1) {
    if (ka >= alen || kb >= blen) break;

    T va = a[ka];
    T vb = b[kb];

    // If you now that you don't have self edges then you don't need to check for them and you can get better performance.
#if(1)
    // Skip self-edges.
    if ((va == ai)) {
      ++ka;

      if (ka >= alen) break;
      va = a[ka];
    }
    if ((vb == bi)) {
      ++kb;
      if (kb >= blen) break;
      vb = b[kb];
    }
#endif
    //
    if (va == vb) {
      ++ka; ++kb; ++out;
    }
    else if (va < vb) {
      // Advance ka
      ++ka;
      while (ka < alen && a[ka] < vb) ++ka;
    } else {
      // Advance kb
      ++kb;
      while (kb < blen && va > b[kb]) ++kb;
    }
  }
	return out;
}

template <typename T>
void count_all_trianglesCPU (const T nv, const T * off,
    const T * ind, int64_t * triNE,
    int64_t* allTriangles)
{
	int64_t edge=0;
	int64_t sum=0;
  for (T src = 0; src < nv; src++)
  {
		T srcLen=off[src+1]-off[src];
		for(T iter=off[src]; iter<off[src+1]; iter++)
		{
			T dest=ind[iter];
			T destLen=off[dest+1]-off[dest];
			triNE[edge]= count_intersectionsReg (src, srcLen, ind+off[src],
													dest, destLen, ind+off[dest]);
			sum+=triNE[edge++];
		}
	}
	*allTriangles=sum;
}

template <typename T>
void count_all_trianglesCPU (const T nv, const std::vector<T> off,
    const std::vector<T> ind, int64_t * triNE,
    int64_t* allTriangles)
{
	int64_t edge=0;
	int64_t sum=0;
  for (T src = 0; src < nv; src++)
  {
		T srcLen=off.at(src+1)-off.at(src);
		for(int iter=off[src]; iter<off[src+1]; iter++)
		{
      T dest=ind.at(iter);
      T destLen=off.at(dest+1)-off.at(dest);
      triNE[edge]= count_intersectionsReg (src, srcLen, ind.data()+off[src],
                          dest, destLen, ind.data()+off[dest]);
      sum+=triNE[edge++];
		}
	}
	*allTriangles=sum;
}

template <typename T>
void count_all_trianglesCPU (const T nv, const std::vector<T> vertexList,
    const std::vector<T> off, const std::vector<T> ind, int64_t * triNE,
    int64_t* allTriangles)
{
	int64_t edge=0;
	int64_t sum=0;
  for (auto src : vertexList)
  {
		T srcLen=off.at(src+1)-off.at(src);
		for(int iter=off[src]; iter<off[src+1]; iter++)
		{
      T dest = 0;
      try
      {
        dest=ind.at(iter);
        T destLen=off.at(dest+1)-off.at(dest);
        triNE[edge]= count_intersectionsReg (src, srcLen, ind.data()+off[src],
            dest, destLen, ind.data()+off[dest]);
        sum+=triNE[edge++];
      }
      catch (const std::out_of_range& oor)
      {
        std::cerr<<"bad access:\t"<<src<<"\t"<<dest<<"\n";
        return;
      }
		}
	}
	*allTriangles=sum;
}

template <typename T>
void count_all_trianglesCPU (const T nv, const T* vertexList,
    const T* off, const T* ind, int64_t * triNE,
    int64_t* allTriangles)
{
	int64_t edge=0;
	int64_t sum=0;
  for(T srcIndex = 0; srcIndex < nv; srcIndex++)
  {
    T src = vertexList[srcIndex];
		T srcLen=off[src+1]-off[src];
		for(int iter=off[src]; iter<off[src+1]; iter++)
		{
      T dest = 0;
      try
      {
        dest=ind[iter];
        T destLen=off[dest+1]-off[dest];
        triNE[edge]= count_intersectionsReg (src, srcLen, ind+off[src],
            dest, destLen, ind+off[dest]);
        sum+=triNE[edge++];
      }
      catch (const std::out_of_range& oor)
      {
        std::cerr<<"bad access:\t"<<src<<"\t"<<dest<<"\n";
        return;
      }
		}
	}
	*allTriangles=sum;
}

template <typename T>
int64_t cpuRun(T nv, T ne, T * off, T * ind, float& timeCPU)
{
  int64_t * triNE = new int64_t[ne]();
  int64_t allTrianglesCPU = 0;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  count_all_trianglesCPU (nv, off,ind, triNE, &allTrianglesCPU);
  end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> elapsed_time = end - start;
  timeCPU = elapsed_time.count();
  //std::cerr<<"CPU : \t"<<allTrianglesCPU<<" \t"<<timeCPU<<"\n";
  delete[] triNE;
  return allTrianglesCPU;
}

template <typename T>
int64_t cpuRun(T nv, T ne, std::vector<T> off, std::vector<T> ind,
    float& timeCPU)
{
  int64_t * triNE = new int64_t[ne];
  int64_t allTrianglesCPU = 0;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  count_all_trianglesCPU (nv, off,ind, triNE, &allTrianglesCPU);
  end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> elapsed_time = end - start;
  timeCPU = elapsed_time.count();
  //std::cerr<<"CPU : \t"<<allTrianglesCPU<<" \t"<<timeCPU<<"\n";
  delete[] triNE;
  return allTrianglesCPU;
}

template <typename T>
int64_t cpuRun(T nv, T ne, std::vector<T> vertexList, std::vector<T> off,
    std::vector<T> ind, float& timeCPU)
{
  int64_t * triNE = new int64_t[ne];
  int64_t allTrianglesCPU = 0;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  count_all_trianglesCPU (nv, vertexList, off, ind, triNE, &allTrianglesCPU);
  end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> elapsed_time = end - start;
  timeCPU = elapsed_time.count();
  //std::cerr<<nv<<" | "<<ne<<" : \t"<<allTrianglesCPU<<" \t"<<timeCPU<<"\n";
  delete[] triNE;
  return allTrianglesCPU;
}

template <typename T>
int64_t cpuRun(T nv, T ne, T* vertexList, T* off,
    T* ind, float& timeCPU)
{
  int64_t * triNE = new int64_t[ne]();
  int64_t allTrianglesCPU = 0;
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  count_all_trianglesCPU (nv, vertexList, off, ind, triNE, &allTrianglesCPU);
  end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> elapsed_time = end - start;
  timeCPU = elapsed_time.count();
  //std::cerr<<nv<<" | "<<ne<<" : \t"<<allTrianglesCPU<<" \t"<<timeCPU<<"\n";
  delete[] triNE;
  return allTrianglesCPU;
}

template int64_t cpuRun<int32_t>(int32_t nv, int32_t ne, int32_t * off,
    int32_t * ind, float& timeCPU);

template int64_t cpuRun<int64_t>(int64_t nv, int64_t ne, int64_t * off,
    int64_t * ind, float& timeCPU);

template int64_t cpuRun<int32_t>(int32_t nv, int32_t ne,
    std::vector<int32_t> off, std::vector<int32_t> ind, float& timeCPU);

template int64_t cpuRun<int64_t>(int64_t nv, int64_t ne,
    std::vector<int64_t> off, std::vector<int64_t> ind, float& timeCPU);

template int64_t cpuRun<int32_t>(int32_t nv, int32_t ne,
    std::vector<int32_t> vertexList, std::vector<int32_t> off,
    std::vector<int32_t> ind, float& timeCPU);

template int64_t cpuRun<int64_t>(int64_t nv, int64_t ne,
    int64_t* vertexList, int64_t* off,
    int64_t* ind, float& timeCPU);

template int64_t cpuRun<int32_t>(int32_t nv, int32_t ne,
    int32_t* vertexList, int32_t* off,
    int32_t* ind, float& timeCPU);

template int64_t cpuRun<int64_t>(int64_t nv, int64_t ne,
    std::vector<int64_t> vertexList, std::vector<int64_t> off,
    std::vector<int64_t> ind, float& timeCPU);

template int64_t count_intersectionsReg<int32_t>(const int32_t ai,
    const int32_t alen, const int32_t * a,
    const int32_t bi, const int32_t blen, const int32_t * b);

template int64_t count_intersectionsReg<int64_t>(const int64_t ai,
    const int64_t alen, const int64_t * a,
    const int64_t bi, const int64_t blen, const int64_t * b);

template void count_all_trianglesCPU<int32_t>(const int32_t nv,
    const int32_t * off, const int32_t * ind, int64_t * triNE,
    int64_t* allTriangles);

template void count_all_trianglesCPU<int64_t>(const int64_t nv,
    const int64_t * off, const int64_t * ind, int64_t * triNE,
    int64_t* allTriangles);

template void count_all_trianglesCPU<int32_t> (const int32_t nv,
    const std::vector<int32_t> off, const std::vector<int32_t> ind,
    int64_t * triNE, int64_t* allTriangles);

template void count_all_trianglesCPU<int64_t> (const int64_t nv,
    const std::vector<int64_t> off, const std::vector<int64_t> ind,
    int64_t * triNE, int64_t* allTriangles);

template void count_all_trianglesCPU<int32_t>(const int32_t nv,
    const std::vector<int32_t> vertexList,
    const std::vector<int32_t> off, const std::vector<int32_t> ind,
    int64_t * triNE, int64_t* allTriangles);

template void count_all_trianglesCPU<int64_t>(const int64_t nv,
    const std::vector<int64_t> vertexList,
    const std::vector<int64_t> off, const std::vector<int64_t> ind,
    int64_t * triNE, int64_t* allTriangles);

template void count_all_trianglesCPU<int32_t>(const int32_t nv,
    const int32_t* vertexList,
    const int32_t* off, const int32_t* ind,
    int64_t * triNE, int64_t* allTriangles);

template void count_all_trianglesCPU<int64_t>(const int64_t nv,
    const int64_t* vertexList,
    const int64_t* off, const int64_t* ind,
    int64_t * triNE, int64_t* allTriangles);
