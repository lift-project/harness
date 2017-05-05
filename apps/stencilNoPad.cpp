// [standard includes]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <typeinfo>
#include <vector>

// [external includes]
#include <opencl_utils.h>

// [local includes]
#include "options.h"
#include "stencilNoPadRun.h"
#include "stencilNoPad_harness.h"

int main(int argc, char *argv[]) {
  OptParser op("Harness for simple stencil without boundary conditions.");

  auto opt_platform = op.addOption<unsigned>(
      {'p', "platform", "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>(
      {'d', "device", "OpenCL device index (default 0).", 0});

  auto opt_size = op.addOption<std::size_t>(
      {'s', "size", "Matrix size (default 1024).", 1024});
  auto opt_transpose = op.addOption<bool>(
      {0, "transpose-in", "Transpose the input matrix before computation.",
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
      {'t', "threaded",
       "Use a separate thread for compilation and execution (default true).",
       true});
  auto opt_force = op.addOption<bool>(
      {'f', "force", "Override cached cross validation files.", false});
  auto opt_clean = op.addOption<bool>(
      {'c', "clean", "Clean temporary files and exit.", false});
  op.parse(argc, argv);

  using namespace std;

  // Option handling
  const size_t N = opt_size->get();
  File::set_size(opt_size->get());
  OpenCL::timeout = opt_timeout->get();

  // temporary files
  std::string gold_file = "/tmp/apart_stencilnopad_gold_" + std::to_string(N);
  std::string grid_file = "/tmp/apart_stencilnopad_grid_" + std::to_string(N);

  if (opt_clean->get()) {
    std::cout << "Cleaning..." << std::endl;
    for (const auto &file : {gold_file, grid_file})
      std::remove(file.data());
    return 0;
  }

  // === Loading CSV file ===
  auto all_run = Csv::init(
      [&](const std::vector<std::string> &values) -> std::shared_ptr<Run> {
        return (
            opt_double->get()
                ? std::shared_ptr<Run>(new StencilNoPadRun<double>(values, N))
                : std::shared_ptr<Run>(new StencilNoPadRun<float>(values, N)));
      });
  if (all_run.size() == 0)
    return 0;

  // === OpenCL init ===
  OpenCL::init(opt_platform->get(), opt_device->get());

  // run the harness
  run_harness(all_run, N, grid_file, gold_file, opt_force->get(),
              opt_threaded->get(), opt_binary->get());
}
