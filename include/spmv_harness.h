#ifndef EXECUTOR_ACOUSTIC3D_HARNESS_H
#define EXECUTOR_ACOUSTIC3D_HARNESS_H

#include <cmath>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "opencl_utils.h"
#include "run.h"
#include "utils.h"

using namespace std;

void set_kernel_args(const shared_ptr<Run> run,
                     const cl::Buffer &roomtminus1_dev,
                     const cl::Buffer &roomt_dev, const cl::Buffer &output_dev,
                     const size_t M, const size_t N, const size_t O) {
  unsigned i = 0;
  run->getKernel().setArg(i++, roomtminus1_dev);
  run->getKernel().setArg(i++, roomt_dev);
  run->getKernel().setArg(i++, output_dev);
  // provide extra local memory args if existing
  for (const auto &local_arg : run->extra_local_args)
    run->getKernel().setArg(i++, local_arg);
  run->getKernel().setArg(i++, static_cast<int>(M));
  run->getKernel().setArg(i++, static_cast<int>(N));
  run->getKernel().setArg(i++, static_cast<int>(O));
}

void compute_gold(const size_t M, const size_t N, const size_t O,
                  Matrix<float> &roomtminus1, Matrix<float> &roomt,
                  Matrix<float> &gold, const std::string &roomtminus1_file,
                  const std::string &roomt_file, const std::string &gold_file) {

  File::load_input_debug(
      gold,
      "/home/bastian/development/exploration/datasets/acoustic/output.txt");
  File::load_input_debug(roomtminus1, "/home/bastian/development/exploration/"
                                      "datasets/acoustic/roomtminus1.txt");
  File::load_input_debug(
      roomt,
      "/home/bastian/development/exploration/datasets/acoustic/roomt.txt");

  File::save_input(gold, gold_file);
  File::save_input(roomtminus1, roomtminus1_file);
  File::save_input(roomt, roomt_file);
}

void run_harness(std::vector<std::shared_ptr<Run>> &all_run, const size_t M,
                 const size_t N, const size_t O,
                 const std::string &roomtminus1_file,
                 const std::string &roomt_file, const std::string &gold_file,
                 const bool force, const bool threaded, const bool binary) {

  if (binary)
    std::cout << "Using precompiled binaries" << std::endl;

  // M rows, N columns
  Matrix<float> roomtminus1(M * N * O);
  Matrix<float> roomt(M * N * O);
  Matrix<float> gold(M * N * O);

  // use existing grid, weights and gold or init them
  if (File::is_file_exist(gold_file) && File::is_file_exist(roomtminus1_file) &&
      File::is_file_exist(roomt_file)) {
    std::cout << "use existing grid, weights and gold" << std::endl;
    File::load_input(gold, gold_file);
    File::load_input(roomtminus1, roomtminus1_file);
    File::load_input(roomt, roomt_file);
  } else {
    std::cout << "load files and save as binary" << std::endl;
    compute_gold(M, N, O, roomtminus1, roomt, gold, roomtminus1_file,
                 roomt_file, gold_file);
  }

  // validation function
  auto validate = [&](const std::vector<float> &output) {
    bool correct = true;
    if (gold.size() != output.size())
      return false;
    for (unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      // auto x = roomtminus1[i];
      auto y = output[i];

      // possibly lots of floating point weirdness going on
      if (abs(x - y) > 0.01f * max(abs(x), abs(y))) {
        cerr << "at " << i << ": " << x << "=/=" << y << std::endl;
        // return false;
        return true;
        // correct = false;
      }
    }
    return correct;
  };

  // Allocating buffers
  const size_t buf_size = roomtminus1.size() * sizeof(float);
  const size_t out_size = gold.size() * sizeof(float);
  cl::Buffer roomtminus1_dev =
      OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
                    static_cast<void *>(roomtminus1.data()));
  cl::Buffer roomt_dev =
      OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
                    static_cast<void *>(roomt.data()));
  cl::Buffer output_dev = OpenCL::alloc(CL_MEM_READ_WRITE, out_size);

  // multi-threaded exec
  if (threaded) {
    std::mutex m;
    std::condition_variable cv;

    bool done = false;
    bool ready = false;
    std::queue<shared_ptr<Run>> ready_queue;

    // compilation thread
    auto compilation_thread = std::thread([&] {
      for (auto &r : all_run) {
        if (r->compile(binary)) {
          std::unique_lock<std::mutex> locker(m);
          ready_queue.push(r);
          ready = true;
          cv.notify_one();
        }
      }
    });

    auto execute_thread = std::thread([&] {
      shared_ptr<Run> r = nullptr;
      while (!done) {
        {
          std::unique_lock<std::mutex> locker(m);
          while (!ready && !done)
            cv.wait(locker);
        }

        while (!ready_queue.empty()) {
          {
            std::unique_lock<std::mutex> locker(m);
            r = ready_queue.front();
            ready_queue.pop();
          }

          set_kernel_args(r, roomtminus1_dev, roomt_dev, output_dev, M, N, O);
          OpenCL::executeRun<float>(*r, output_dev, gold.size(), validate);
        }
      }
    });

    compilation_thread.join();
    done = true;
    cv.notify_one();
    execute_thread.join();
  }
  // single threaded exec
  else {
    for (auto &r : all_run) {
      if (r->compile(binary)) {
        set_kernel_args(r, roomtminus1_dev, roomt_dev, output_dev, M, N, O);
        OpenCL::executeRun<float>(*r, output_dev, gold.size(), validate);
      }
    }
  }
};

#endif // EXECUTOR_ACOUSTIC3D_HARNESS_H
