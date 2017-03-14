//
// Created by Bastian Hagedorn
//

#ifndef EXECUTOR_3DRUN_H
#define EXECUTOR_3DRUN_H

#include <cstddef>

#include "run.h"

struct Run3D : public Run {
	// input matrix size
	size_t M;
	size_t N;
	size_t O;

	/**
	 * Deserialize a line from the CSV
	 */
	Run3D(const std::vector<std::string> &values, size_t M, size_t N, size_t O,
			 size_t default_local_0 = 1, size_t default_local_1 = 1,
			 size_t default_local_2 = 1)
	    : Run(values, default_local_0, default_local_1, default_local_2), M(M), N(N),O(O) {}

	void setup(cl::Context context) override {
		// Allocate extra buffers
		/*
	       for (auto &size : extra_buffer_size) {
		       extra_args.push_back({context, CL_MEM_READ_WRITE,
	       (size_t)size});
	       }
	       */

		// Skip the first 3 to compensate for the csv (forgot a
		// drop(3) in scala)
		// for (unsigned i = 0; i < extra_args.size(); ++i)
		// kernel.setArg(3 + i, extra_args[i]);

		/*
		for (unsigned i = 0; i < extra_local_args.size(); ++i)
			kernel.setArg((unsigned)extra_args.size() + 3 + i,
		extra_local_args[i]);

		kernel.setArg((unsigned)extra_local_args.size() +
		(unsigned)extra_args.size() + 3,
			      (int)size);
		kernel.setArg((unsigned)extra_local_args.size() +
		(unsigned)extra_args.size() + 4,
			      (int)size);
						*/
	}
};

#endif // EXECUTOR_3DRUN_H
