//
// Created by s1042579 on 17/08/16.
//

// [standard includes]
#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// [external includes]
#include <opencl_utils.h>

// [local includes]
#include "options.h"
#include "kmeansrun.h"
#include "kmeans_harness.h"

int main(int argc, char *argv[]) {
  OptParser op("Harness for kmeans.");

  // OpenCL options
  auto opt_platform = op.addOption<unsigned>({'p', "platform",
      "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>({'d', "device",
      "OpenCL device index (default 0).", 0});


  // Common options
  auto opt_binary = op.addOption<bool>({'b', "binary", "Load programs as binaries instead of compiling OpenCL-C source.", false});
  auto opt_timeout = op.addOption<float>({'t', "timeout", "Timeout to avoid multiple executions (default 100ms).", 100.0f});
  auto opt_double = op.addOption<bool>({0, "double", "Use double precision.", false});
  auto opt_threaded = op.addOption<bool>({0, "threaded", "Use a separate thread for compilation and execution (default true).", true});
  auto opt_force = op.addOption<bool>({'f', "force", "Override cached cross validation files.", false});
  auto opt_clean = op.addOption<bool>({'c', "clean", "Clean temporary files and exit.", false});

  auto opt_iterations = op.addOption<int>({'i', "iterations",
      "The number of iterations for each experiment (default 10)", 10});
  auto opt_local_combinations = op.addOption<bool>({'l', "local-combinations",
      "Run different valid combinations of local sizes instead of letting the implementation choose if the local size is marked '?'.", false});
  auto opt_local_0 = op.addOption<std::size_t>({0, "l0",
      "Local size in dim 0 to use if specified as '?'", 0});
  auto opt_local_1 = op.addOption<std::size_t>({0, "l1",
      "Local size in dim 1 to use if specified as '?'", 0});
  auto opt_local_2 = op.addOption<std::size_t>({0, "l2",
      "Local size in dim 2 to use if specified as '?'", 0});
  auto opt_min_local_size = op.addOption<std::size_t>({0, "min-local",
      "The minimum local size to use when running the experiments (defaults 1).", 1});


  // K-Means options
  auto opt_num_points = op.addOption<std::size_t>({'s', "points",
      "Matrix size (default 1024).", 1024});
  auto opt_num_clusters = op.addOption<std::size_t>({'c', "clusters",
      "Number of clusters (default 5).", 5});
  auto opt_num_features = op.addOption<std::size_t>({'f', "features",
  "Number of features (default 34).", 34});

  auto opt_dont_transpose_points = op.addOption<bool>({'t', "no-transpose",
      "Don't points before passing to kernel", false});

  op.parse(argc, argv);

  using namespace std;

  // Option handling
  const auto num_points = opt_num_points->get();
  const auto num_clusters = opt_num_clusters->get();
  const auto num_features = opt_num_features->get();

  const auto transpose_points = !opt_dont_transpose_points->get();

  auto local_0 = opt_local_0->get();
  auto local_1 = opt_local_1->get();
  auto local_2 = opt_local_2->get();

  auto size_string = to_string(num_points);

  File::set_size(size_string);
  OpenCL::timeout = opt_timeout->get();
  OpenCL::local_combinations = opt_local_combinations->get();
  OpenCL::min_local_size = opt_min_local_size->get();
  OpenCL::iterations = opt_iterations->get();

  // Result files
  auto gold_file = "/tmp/apart_kmeans_gold_" + size_string;
  auto points_file = "/tmp/apart_kmeans_points_" + size_string;
  auto clusters_file = "/tmp/apart_kmeans_clusters_" + size_string;

  if(*opt_clean) {
    cout << "Cleaning..." << endl;
    for(const auto& file: {gold_file, points_file, clusters_file})
      remove(file.data());
    return 0;
  }

  // === Loading CSV file ===
  auto all_run = Csv::init(
      [&](const std::vector<std::string>& values) -> std::shared_ptr<Run> {
        return (opt_double->get() ?

                std::shared_ptr<Run>(new KMeansRun<double>(values,
                    num_points, num_clusters, num_features,
                    local_0, local_1, local_2)) :
                std::shared_ptr<Run>(new KMeansRun<float>(values,
                    num_points, num_clusters, num_features,
                    local_0, local_1, local_2)));
      });

  if (all_run.size() == 0) return 0;

  // === OpenCL init ===
  OpenCL::init(opt_platform->get(), opt_device->get());

  // run the harness
  if (opt_double->get())
    run_harness<double>(
        all_run, num_points, num_clusters, num_features,
        points_file, clusters_file, gold_file,
        opt_force->get(),
        opt_threaded->get(), opt_binary->get(), transpose_points
    );
  else
    run_harness<float>(
    all_run, num_points, num_clusters, num_features,
        points_file, clusters_file, gold_file,
        opt_force->get(),
        opt_threaded->get(), opt_binary->get(), transpose_points
    );
}
