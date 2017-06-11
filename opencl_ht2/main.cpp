#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("pref_sum.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);

      // create a message to send to kernel
      size_t const block_size = 256;
      size_t const N;
      std::cin >> N;

      size_t input_size = N - N % block_size + block_size;
      std::vector<float> input(input_size, 0);
      for(int i = 0; i < N; ++i) {
          std::cin >> input[i];
      }

      // allocate device buffer to hold message
      cl::Buffer dev_input (context, CL_MEM_READ_WRITE, sizeof(float) * input_size);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(int) * input_size, &input[0]);
      queue.finish();

      // load named kernel from opencl source
      cl::Kernel kernel_hs(program, "scan_hillis_steele");
      cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(input_size), cl::NDRange(block_size));

      cl::Kernel kernel_pr(program, "propagate");
      cl::KernelFunctor back_pr(kernel_pr, queue, cl::NullRange, cl::NDRange(input_size), cl::NDRange(block_size));

      uint shift = 1;
      for(; shift < (uint)N; shift *= block_size) {
          cl::Event event = scan_hs(dev_input, cl::__local(sizeof(float) * block_size), cl::__local(sizeof(int) * block_size), shift, N);
          event.wait();
      }
      shift /= (block_size * block_size);
      for(; shift > 1; shift /= block_size) {
          cl::Event event = back_pr(dev_input, cl::__local(sizeof(float) * block_size), shift, N);
          event.wait();
      }

//      cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
//      cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
//      cl_ulong elapsed_time = end_time - start_time;

      queue.enqueueReadBuffer(dev_input, CL_TRUE, 0, sizeof(int) * input_size, &input[0]);
      for(int i = 0; i < N; ++i) {
          std::cout << std::fixed << std::setprecision(3) << input[i] << " ";
      }
      std::cout  << std::endl;
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
