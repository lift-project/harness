#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "csv_utils.h"

// Object representation of the run
class Run {
      public:
	// global range
	std::size_t glob1 = 0;
	std::size_t glob2 = 0;
	std::size_t glob3 = 0;

	// local range
	std::size_t loc1 = 0;
	std::size_t loc2 = 0;
	std::size_t loc3 = 0;

	// Defaults in case not specified
	std::size_t default_local_0;
	std::size_t default_local_1;
	std::size_t default_local_2;

	// list of additional buffers to allocate
	std::vector<int> extra_buffer_size;
	std::vector<cl::Buffer> extra_args;

	// list of additional local buffers to allocate
	std::vector<cl::LocalSpaceArg> extra_local_args;
	std::size_t sum_local = 0;

	// hash file
	std::string hash;

	// compiled kernel
	cl::Kernel kernel;

	Run(const std::vector<std::string> &values, std::size_t default_local_0,
	    std::size_t default_local_1, std::size_t default_local_2);

	// Load the file and compile the program
	bool compile(bool binary_mode);

	virtual ~Run() {}

	virtual void setup(cl::Context context) = 0;

	virtual void cleanup() {
		extra_buffer_size.clear();
		kernel = cl::Kernel();
	}

	cl::Kernel &getKernel() { return kernel; }
};
