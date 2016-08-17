//
// Created by s1042579 on 17/08/16.
//

#ifndef EXECUTOR_NBODYRUN_H
#define EXECUTOR_NBODYRUN_H

#include "run.h"

template<typename T>
struct NBodyRun: public Run {

  std::size_t size;

  /**
   * Deserialize a line from the CSV
   */
  NBodyRun(const std::vector<std::string>& values, std::size_t size,
      std::size_t default_local_0,
      std::size_t default_local_1, std::size_t default_local_2):
      Run(values, default_local_0, default_local_1, default_local_2),
      size(size) {}

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for(auto &size: extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

    cl_uint idx = 5;

    // Skip the first 5 to compensate for the other arguments
    for(const auto &arg: extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local: extra_local_args)
      kernel.setArg(idx++, local);

    kernel.setArg(idx, (int) size);
  }

};

#endif //EXECUTOR_NBODYRUN_H
