//
// Created by s1042579 on 17/08/16.
//

#ifndef EXECUTOR_KMEANSRUN_H
#define EXECUTOR_KMEANSRUN_H

#include "run.h"

template<typename T>
struct KMeansRun: public Run {

  std::size_t num_points;
  std::size_t num_clusters;
  std::size_t num_features;

  /**
   * Deserialize a line from the CSV
   */
  KMeansRun(const std::vector<std::string>& values,
      std::size_t num_points, std::size_t num_clusters, std::size_t num_features,
      std::size_t default_local_0, std::size_t default_local_1, std::size_t default_local_2):
      Run(values, default_local_0, default_local_1, default_local_2),
      num_points(num_points), num_clusters(num_clusters), num_features(num_features) {}

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for(auto &size: extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

    cl_uint idx = 3;

    // Skip the first 5 to compensate for the other arguments
    for(const auto &arg: extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local: extra_local_args)
      kernel.setArg(idx++, local);

    kernel.setArg(idx++, (int) num_clusters);
    kernel.setArg(idx++, (int) num_features);
    kernel.setArg(idx, (int) num_points);
  }

};

#endif //EXECUTOR_KMEANSRUN_H
