// [standard includes]
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

// [external includes]
#include <opencl_utils.h>

// [local includes]
#include "options.h"
#include "mmrun.h"
#include "mm_harness.h"

int main(int argc, char *argv[]) {
  OptParser op("Harness for simple matrix-matrix multiply.");

  auto opt_platform = op.addOption<unsigned>(
      {'p', "platform", "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>(
      {'d', "device", "OpenCL device index (default 0).", 0});

  auto opt_size = op.addOption<std::size_t>(
      {'s', "size", "Matrix size (default 1024).", 1024});
  auto opt_size_k = op.addOption<std::size_t>(
      {'k', "size-k", "Matrix size in the common dimension (default 1024).",
       1024});
  auto opt_size_m = op.addOption<std::size_t>(
      {'m', "size-m",
       "Size of matrix A in the non-common dimension (default 1024).", 1024});
  auto opt_size_n = op.addOption<std::size_t>(
      {'n', "size-n",
       "Size of matrix B in the non-common dimension (default 1024).", 1024});
  auto opt_transposeA = op.addOption<bool>(
      {0, "transpose-A", "Transpose the first matrix before computation.",
       false});
  auto opt_transposeB = op.addOption<bool>(
      {0, "transpose-B", "Transpose the second matrix before computation.",
       false});
  auto opt_transposeRes = op.addOption<bool>(
      {0, "transpose-res", "Transpose the output before cross validation.",
       false});

  auto opt_binary = op.addOption<bool>(
      {'b', "binary",
       "Load programs as binaries instead of compiling OpenCL-C source.",
       false});
  auto opt_timeout = op.addOption<float>(
      {'t', "timeout", "Timeout to avoid multiple executions (default 100ms).",
       100.0f});
  auto opt_double =
      op.addOption<bool>({0, "double", "Use double precision.", false});
  auto opt_threaded = op.addOption<bool>(
      {0, "threaded",
       "Use a separate thread for compilation and execution (default true).",
       true});
  auto opt_force = op.addOption<bool>(
      {'f', "force", "Override cached cross validation files.", false});
  auto opt_clean = op.addOption<bool>(
      {'c', "clean", "Clean temporary files and exit.", false});

  auto opt_iterations = op.addOption<int>(
      {'i', "iterations",
       "The number of iterations for each experiment (default 10)", 10});
  auto opt_local_combinations = op.addOption<bool>(
      {'l', "local-combinations", "Run different valid combinations of local "
                                  "sizes instead of letting the implementation "
                                  "choose if the local size is marked '?'.",
       false});
  auto opt_local_0 = op.addOption<std::size_t>(
      {0, "l0", "Local size in dim 0 to use if specified as '?'", 0});
  auto opt_local_1 = op.addOption<std::size_t>(
      {0, "l1", "Local size in dim 1 to use if specified as '?'", 0});
  auto opt_local_2 = op.addOption<std::size_t>(
      {0, "l2", "Local size in dim 2 to use if specified as '?'", 0});
  auto opt_min_local_size = op.addOption<std::size_t>(
      {0, "min-local", "The minimum local size to use when running the "
                       "experiments (defaults 1).",
       1});

  op.parse(argc, argv);

  using namespace std;

  // Option handling
  const size_t common_size = opt_size->get();
  size_t N = opt_size_n->get();
  size_t M = opt_size_m->get();
  size_t K = opt_size_k->get();

  size_t local_0 = opt_local_0->get();
  size_t local_1 = opt_local_1->get();
  size_t local_2 = opt_local_2->get();

  auto size_string = to_string(K) + "_" + to_string(M) + "_" + to_string(N);

  if (M == 1024 && K == 1024 && N == 1024) {
    N = common_size;
    K = common_size;
    M = common_size;
    size_string = to_string(common_size);
  }

  File::set_size(size_string);
  OpenCL::timeout = opt_timeout->get();
  OpenCL::local_combinations = opt_local_combinations->get();
  OpenCL::min_local_size = opt_min_local_size->get();
  OpenCL::iterations = opt_iterations->get();

  // Result files
  auto gold_file = "/tmp/apart_mm_gold_" + size_string;
  auto matA_file = "/tmp/apart_mm_A_" + size_string;
  auto matB_file = "/tmp/apart_mm_B_" + size_string;

  if (*opt_clean) {
    cout << "Cleaning..." << endl;
    for (const auto &file : {gold_file, matA_file, matB_file})
      remove(file.data());
    return 0;
  }

  // === Loading CSV file ===
  auto all_run = Csv::init(
      [&](const std::vector<std::string> &values) -> std::shared_ptr<Run> {
        return (opt_double->get()
                    ?

                    std::shared_ptr<Run>(new MMRun<double>(
                        values, M, K, N, local_0, local_1, local_2))
                    : std::shared_ptr<Run>(new MMRun<float>(
                          values, M, K, N, local_0, local_1, local_2)));
      });

  if (all_run.size() == 0)
    return 0;

  // === OpenCL init ===
  OpenCL::init(opt_platform->get(), opt_device->get());

  // run the harness
  if (opt_double->get())
    run_harness<double>(all_run, M, K, N, matA_file, matB_file, gold_file,
                        opt_force->get(), opt_transposeA->get(),
                        opt_transposeB->get(), opt_transposeRes->get(),
                        opt_threaded->get(), opt_binary->get());
  else
    run_harness<float>(all_run, M, K, N, matA_file, matB_file, gold_file,
                       opt_force->get(), opt_transposeA->get(),
                       opt_transposeB->get(), opt_transposeRes->get(),
                       opt_threaded->get(), opt_binary->get());
}
