#ifndef RUN_SPMV_H
#define RUN_SPMV_H

#include "run.h"

class SPMVRun : public Run {
  size_t num_args;
  vector<int> &size_arguments;

public:
  /**
   * Deserialize a line from the CSV
   */
  SPMVRun(const std::vector<std::string> &values, vector<int> &size_arguments,
          size_t num_args, size_t default_local_0 = 1,
          size_t default_local_1 = 1, size_t default_local_2 = 1)
      : Run(values, default_local_0, default_local_1, default_local_2),
        num_args(num_args), size_arguments(size_arguments) {}

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for (auto &size : extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

    auto idx = (cl_uint)num_args + 1;

    // Skip the first num_args to account for inputs/outputs
    for (const auto &arg : extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local : extra_local_args)
      kernel.setArg(idx++, local);

    for (const auto &size : size_arguments)
      kernel.setArg(idx++, size);
  }
};

#endif // RUN_SPMV_H