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
#include "kernel.h"
#include "options.h"
#include "run.h"
#include "sparse_matrix.hpp"
#include "spmv_harness.h"
#include "spmvrun.h"

int main(int argc, char *argv[]) {
  OptParser op(
      "Harness for SPMV sparse matrix dense vector multiplication benchmarks");

  auto opt_platform = op.addOption<unsigned>(
      {'p', "platform", "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>(
      {'d', "device", "OpenCL device index (default 0).", 0});
  auto opt_iterations = op.addOption<unsigned>(
      {'i', "iterations",
       "Execute each kernel 'iterations' times (default 10).", 10});

  //   auto opt_input_file = op.addOption<std::string>({'f', "file", "Input
  //   file"});

  auto opt_matrix_file =
      op.addOption<std::string>({'m', "matrix", "Input matrix"});
  auto opt_kernel_file =
      op.addOption<std::string>({'k', "kernel", "Input kernel"});
  auto opt_run_file =
      op.addOption<std::string>({'r', "runfile", "Run configuration file"});

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

  const std::string matrix_filename = opt_matrix_file->get();
  const std::string kernel_filename = opt_kernel_file->get();

  std::cout << "matrix_filename " << matrix_filename << std::endl;
  std::cout << "kernel_filename " << kernel_filename << std::endl;

  SparseMatrix<float> matrix(matrix_filename);
  Kernel<float> kernel(kernel_filename);

  if (matrix.height() != matrix.width()) {
    std::cout << "Matrix is not square. Failing computation." << std::endl;
    std::cerr << "Matrix is not square. Failing computation." << std::endl;
    std::exit(2);
  }

  // specialise the matrix for the kernel given
  auto cl_matrix = kernel.specialiseMatrix(matrix, 0.0f);

  // extract size variables from it
  int v_Height_cl = cl_matrix.getCLVHeight();
  int v_Width_cl = cl_matrix.getCLVWidth();
  int v_Length_cl = cl_matrix.rows;

  std::vector<int> size_args{v_Height_cl, v_Width_cl, v_Length_cl};

  //   // === Loading exec CSV file ===
  std::vector<std::shared_ptr<Run>> all_run = Csv::init(
      [&](const std::vector<std::string> &values) -> std::shared_ptr<Run> {

        auto run = new SPMVRun(values, size_args, kernel.getArgs().size());
        return std::shared_ptr<Run>(run);

      });
  //   if (all_run.size() == 0)
  //     return 0;

  //   // === OpenCL init ===
  OpenCL::timeout = opt_timeout->get();
  OpenCL::init(opt_platform->get(), opt_device->get(), opt_iterations->get());

  //   // run the harness

  //   run_harness(all_run, M, N, O, roomtminus1_file, roomt_file, gold_file,
  //               opt_force->get(), opt_threaded->get(), opt_binary->get());
}
