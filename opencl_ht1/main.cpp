#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

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
      cl::CommandQueue queue(context, devices[0]);

      // load input data
      int n;
      int m;
      std::vector<float> a;
      std::vector<float> b;
      std::ifstream in("input.txt");
      in >> n >> m;
      a.resize(n * n);
      b.resize(m * m);
      for(int i = 0; i < n; ++i) {
          for(int j = 0; j < n; ++j) {
              in >> a[i * n + j];
          }
      }
      for(int i = 0; i < m; ++i) {
          for(int j = 0; j < m; ++j) {
              in >> b[i * m + j];
          }
      }
      // prepare for output
      std::ofstream out("output.txt");

      // load opencl source
      std::ifstream cl_file("matrix_convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
	  size_t const block_size = 16;
      program.build(devices, "-D BLOCK_SIZE=16");

//      // create a message to send to kernel
//      size_t const N = 256;
//      size_t const matrix_size = N * N;

//      int a[matrix_size];
//      int b[matrix_size];
//      int c[matrix_size];
//      for (size_t i = 0; i < N; ++i)
//      {
//         for (size_t j = 0; j < N; ++j)
//         {
//            size_t idx = i * N + j;
//            a[idx] = rand() % 10;
//            b[idx] = rand() % 10;
//            c[idx] = 0;
//         }
//      }

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * n * n);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(float) * m * m);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n * n, a.data());
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * m * m, b.data());

      // load named kernel from opencl source
      cl::Kernel kernel(program, "matrix_conv");
      cl::KernelFunctor matrix_mult(kernel, queue, cl::NullRange, cl::NDRange(n, n), cl::NDRange(block_size, block_size));
      matrix_mult(dev_a, dev_b, dev_c, n);

//      cl::Kernel kernel_shared(program, "matrix_mult_shared");
//      cl::KernelFunctor matrix_mult_shared(kernel_shared, queue, cl::NullRange,
//		                                   cl::NDRange(N, N), cl::NDRange(block_size, block_size));
	  //matrix_mult_shared(dev_a, dev_b, dev_c, (int)N);

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * n * n, c);

      for (size_t i = 0; i < n; ++i)
      {
         for (size_t j = 0; j < n; ++j)
         {
            size_t idx = i * n + j;
            out << c[idx] << " ";
         }
         out << std::endl;
      }
      out << std::endl;

//      for (size_t i = 0; i < N; ++i)
//      {
//         for (size_t j = 0; j < N; ++j)
//         {
//            size_t idx = i * N + j;

//            int sum = 0;
//            for (int k = 0; k < N; ++k)
//               sum += a[i * N + k] * b[k * N + j];
//			if (c[idx] != sum)
//				std::cout << i << " " << j << std::endl;
//            //std::cout << c[idx] - sum << " ";
//         }
//         //std::cout << std::endl;
//      }
//      std::cout << "finished" << std::endl;
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
