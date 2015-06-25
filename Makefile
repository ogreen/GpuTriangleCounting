ARCH=sm_35
COMPILE = nvcc -O3 -arch=$(ARCH) -std=c++11 #$(DBG_FLAGS)
EXEC = bin/ccgpunew
CPUTRIANGLECOUNT = bin/ofiles/cputrianglecount.o
GPUTRIANGLECOUNT = bin/ofiles/gputrianglecount.o
CPURUN = bin/ofiles/cpurun.o
GPURUN = bin/ofiles/gpurun.o
MAIN = bin/ofiles/main.o
PARAM = bin/ofiles/param.o
#GRAPHOP = bin/ofiles/graphOperations.o
GRAPHREAD = bin/ofiles/graphRead.o
DBG_FLAGS = -lineinfo -g -G

CPURUNDEPS = src/cpurun.cpp include/cpurun.hpp
GPURUNDEPS = src/gpurun.cu include/gpurun.cuh
GRAPHREADDEPS = src/graphRead.cpp include/graphRead.hpp
#GRAPHOPDEPS = src/graphOperations.cpp include/graphOperations.hpp
GPUTRIANGLECOUNTDEPS = src/clusteringCount.cu include/clusteringCount.cuh src/reduce.cu include/reduce.cuh
CPUTRIANGLECOUNTDEPS = src/clusteringCount.cpp include/clusteringCount.hpp
PARAMDEPS = src/param.cu include/param.cuh
#MAINDEPS = src/main.cu include/main.cuh include/results.cuh $(GRAPHOP) $(CPURUN) $(CPUTRIANGLECOUNT) $(GPURUN) $(GPUTRIANGLECOUNT) $(GRAPHREAD)
MAINDEPS = src/main.cu include/main.cuh include/results.cuh $(GPURUN) $(GPUTRIANGLECOUNT) $(GRAPHREAD)
#EXECDEPS = $(MAIN) $(PARAM) $(CPURUN) $(CPUTRIANGLECOUNT) $(GPURUN) $(GPUTRIANGLECOUNT) $(GRAPHREAD) $(GRAPHOP)
#EXECDEPS = $(MAIN) $(PARAM)  $(GPURUN) $(GPUTRIANGLECOUNT) $(GRAPHREAD) 
EXECDEPS = $(MAIN) $(PARAM) $(GPURUN)  $(GPUTRIANGLECOUNT) $(GRAPHREAD) 

all: $(EXEC)

$(MAIN): $(MAINDEPS)
	$(COMPILE) -c src/main.cu -o $(MAIN)

#$(CPURUN): $(CPURUNDEPS)
#	$(COMPILE) -c src/cpurun.cpp -o $(CPURUN)

$(GPURUN): $(GPURUNDEPS)
	$(COMPILE) -c src/gpurun.cu -Xcompiler -fopenmp -o $(GPURUN)

$(PARAM): $(PARAMDEPS)
	$(COMPILE) -c src/param.cu -o $(PARAM)

#$(GRAPHOP): $(GRAPHOPDEPS)
#	$(COMPILE) -c src/graphOperations.cpp -Xcompiler -fopenmp -lgomp -o $(GRAPHOP)

$(GRAPHREAD): $(GRAPHREADDEPS)
	$(COMPILE) -c src/graphRead.cpp -o $(GRAPHREAD)

$(GPUTRIANGLECOUNT): $(GPUTRIANGLECOUNTDEPS)
	$(COMPILE) -c src/clusteringCount.cu -o $(GPUTRIANGLECOUNT)

#$(CPUTRIANGLECOUNT): $(CPUTRIANGLECOUNTDEPS)
#	$(COMPILE) -c src/clusteringCount.cpp -o $(CPUTRIANGLECOUNT)

$(EXEC): $(EXECDEPS)
	$(COMPILE) $(EXECDEPS) -lnvToolsExt -Xcompiler -fopenmp -o $(EXEC)
	g++ -std=c++11 src/csrdatacreate.cpp -fopenmp 

clean:
	rm -f $(EXEC)
	rm -f bin/csr
	rm -f bin/ofiles/*
