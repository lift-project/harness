#ifndef EXECUTOR_MM_H
#define EXECUTOR_MM_H

#include "run.h"

template <typename T> struct MMRun : public Run {

  std::size_t M;
  std::size_t K;
  std::size_t N;

  /**
   * Deserialize a line from the CSV
   */
  MMRun(const std::vector<std::string> &values, std::size_t M, std::size_t K,
        std::size_t N, std::size_t default_local_0, std::size_t default_local_1,
        std::size_t default_local_2)
      : Run(values, default_local_0, default_local_1, default_local_2), M(M),
        K(K), N(N) {}

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for (auto &size : extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

    cl_uint idx = 3;

    // Skip the first 3 to compensate for the first 3 arguments
    for (const auto &arg : extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local : extra_local_args)
      kernel.setArg(idx++, local);

    // Arguments in alphabetic order
    kernel.setArg(idx++, (int)K);
    kernel.setArg(idx++, (int)M);
    kernel.setArg(idx, (int)N);
  }
};

#endif // EXECUTOR_MM_H
