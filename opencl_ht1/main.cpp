#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <string>

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
      std::vector<float> c;
      std::ifstream in("input.txt");
      in >> n >> m;
      a.resize(n * n);
      b.resize(m * m);
      c.resize(n * n, 0);
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
      size_t block_size = 16;
      while(block_size > n || n % block_size != 0) {
	  block_size /= 2;
      }
      std::stringstream ss;
      ss << block_size;
      std::string arg =  "-D BLOCK_SIZE=" + ss.str();
      program.build(devices, arg.data());

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * n * n);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(float) * m * m);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * n * n, a.data());
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * m * m, b.data());

      // load named kernel from opencl source
      cl::Kernel kernel(program, "matrix_conv");
      
      cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange, cl::NDRange(n, n), cl::NDRange(block_size, block_size));
      matrix_conv(dev_a, dev_b, dev_c, n, m);

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * n * n, c.data());

/*
      for (size_t i = 0; i < n; ++i)
      {
         for (size_t j = 0; j < n; ++j)
         {
            size_t idx = i * n + j;
            std::cerr << a[idx] << " ";
         }
         std::cerr << std::endl;
      }
      std::cerr << std::endl;

      std::cerr << m << std::endl;
      for (size_t i = 0; i < m; ++i)
      {
         for (size_t j = 0; j < m; ++j)
         {
            size_t idx = i * m + j;
            std::cerr << b[idx] << " ";
         }
         std::cerr << std::endl;
      }
      std::cerr << std::endl;
*/
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
/*
      int hm = (m - 1) / 2;
      for(int row = 0; row < n; ++row) {
	  for(int col = 0; col < n; ++col) { 
   	      float sum = 0;
              for(int k = -hm; k <= hm; ++k) {
        	  for(int l = -hm; l <= hm; ++l) {
            	      if(row + k >= 0 && row + k < n
                	&& col + l >= 0 && col + l < n) {
                          sum += a[(row + k) * n + (col + l)]
                                  * b[(k + hm) * m + (l + hm)];
		      }
                  }
              }
   	      c[row * n + col] = sum;
	      std::cerr << c[row * n + col] << " ";
          }
          std::cerr << std::endl;
      }
      std::cerr << std::endl;
*/
   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
