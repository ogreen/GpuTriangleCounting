#include "../include/main.cuh"

template <typename T>
void run(int argc, char * argv[])
{
  Param param = paramProcess(argc, argv);
  if(!param.valid)
  {
    std::cerr<<"Exiting..\n";
    exit(0);
  }
      if(!param.testAll)
      {
        singleParamTestGPURun<T>(param);
      }
      else
      {
        allParamTestGPURun<T>(param);
      }

      if((param.deviceMap).size() > 0)
      {
        cudaSetDevice((param.deviceMap).at(0));
        std::cerr<<"\ndeviceMap provided\n";
      }
      else
      {
        cudaSetDevice(0);
      }


}

int main(int argc, char * argv[])
{
  run<int64_t>(argc, argv);
  return 0;
}
