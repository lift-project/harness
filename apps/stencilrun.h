//
// Created by Bastian Hagedorn
//

#ifndef EXECUTOR_STENCILRUN_H
#define EXECUTOR_STENCILRUN_H

#include <cstddef>

#include "run.h"

template <typename T> struct StencilRun : public Run {
	// input matrix size
	std::size_t size;

	/**
	 * Deserialize a line from the CSV
	 */
	StencilRun(const std::vector<std::string> &values, size_t size, size_t default_local_0 = 1,
		   size_t default_local_1 = 1, size_t default_local_2 = 1)
	    : Run(values, default_local_0, default_local_1, default_local_2), size(size) {}

	void setup(cl::Context context) override {
		// Allocate extra buffers
		for (auto &size : extra_buffer_size)
			extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

		/*
		// Skip the first 3 to compensate for the csv (forgot a drop(3) in scala)
		for (unsigned i = 0; i < extra_args.size(); ++i)
			kernel.setArg(3 + i, extra_args[i]);

		for (unsigned i = 0; i < extra_local_args.size(); ++i)
			kernel.setArg((unsigned)extra_args.size() + 3 + i, extra_local_args[i]);

		kernel.setArg((unsigned)extra_local_args.size() + (unsigned)extra_args.size() + 3,
			      (int)size);
		kernel.setArg((unsigned)extra_local_args.size() + (unsigned)extra_args.size() + 4,
			      (int)size);
						*/
	}
};

#endif // EXECUTOR_STENCILRUN_H
