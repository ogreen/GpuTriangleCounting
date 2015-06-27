# GpuTriangleCounting
In this repo you can find an algorithm for triangle counting on the GPU using CUDA.

Specifically, this implementation computes the number of triangles for each vertex, this is equivalent to computing the "local clustering coefficient" value. This is slightly different than computing a single value for the entire graph, aka "global clustering coeffcient". The key difference between these approaches is in the book keeping - the global value is easier for updating than the local values and thus has better performance. 

The following explains how to get a 2x performance increase (from the local-computation to the global-computation), make the following change (DISCLAIMER: I did this a while back and it may be wrong) :
* Go to clusteringCount.cu
* Go to function "count_all_trianglesGPU".
* And add a comparison of "if (dest<src) break"
* Multiply the number of total number of triangles by 2.


For additional information on the algorithm, I recommend reading the following papers:
* "Fast Triangle Counting on the GPU" - contains detailed information on the paper.
* "GPU Merge Path: A GPU Merging Algorithm" - The GPU version of Merge Path. Includes a detailed discussion of the multi-level  partitioning required for performance on the GPU.
* "Merge Path-Parallel Merging Made Simple" - introduces the Merge Path concept which is a key component of the GPU triangle counting algorithm.

If you are interested on additional optimizations for triangle counting for the CPU see:

* "Load Balanced Clustering Coefficients" - explains how to design load-balanced algorithms for triangle counting that gets good performance for power law graphs and for large thread counts.
* "Faster Clustering Coefficient Using Vertex Covers" - this algorithm reduces the number of interesection required for counting triangles.


To build:
* Clone the repo.
* cd in repo.
* mkdir bin
* cd bin && mkdir ofile
* cd ..
* make

To execute:
* cd into repo
* Use graphs from the DIMACS 10 challange.



