//
// Created by Bastian Hagedorn
//

#ifndef EXECUTOR_STENCIL_HARNESS_H
#define EXECUTOR_STENCIL_HARNESS_H

#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "opencl_utils.h"
#include "run.h"
#include "utils.h"

/**
 * FIXME: This is a lazy copy paste of the old main with a template switch for
 * single and double
 * precision
 */
void run_harness(std::vector<std::shared_ptr<Run>> &all_run, const size_t N,
                 const std::string &grid_file, const std::string &gold_file,
                 const bool force, const bool threaded, const bool binary) {
  using namespace std;

  if (binary)
    std::cout << "Using precompiled binaries" << std::endl;
  // Compute input and output
  Matrix<float> grid(N * N);
  Matrix<float> gold(N * N);

  if (File::is_file_exist(gold_file) && File::is_file_exist(grid_file)) {
    File::load_input(gold, gold_file);
    File::load_input(grid, grid_file);
  } else {
    for (unsigned y = 0; y < N; ++y) {
      for (unsigned x = 0; x < N; ++x) {
        grid[y * N + x] = (((y * 3 + x * 2) % 10) + 1) * 1.0f;
      }
    }

    // compute gold
    for (unsigned y = 0; y < N; y++) {
      for (unsigned x = 0; x < N; x++) {
        float sum = 0.0f;
        for (unsigned i = -1; i < 2; i++) {
          for (unsigned j = -1; j < 2; j++) {
            int posX = (x + i);
            int posY = (y + j);
            // clamp boundary
            // todo fix
            // posX = std::min(posX, N);
            // posY = std::min(posY, N);
            // posX = std::max(posX, 0);
            // posY = std::max(posY, 0);

            sum += grid[posY * N + posX];
          }
        }
        gold[y * N + x] = sum;
      }
    }

    File::save_input(gold, gold_file);
    File::save_input(grid, grid_file);
  }

  // validation function
  auto validate = [&](const std::vector<float> &output) {
    if (gold.size() != output.size())
      return false;
    for (unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      auto y = output[i];

      if (abs(x - y) > 0.001f * max(abs(x), abs(y))) {
        cout << "at " << i << ": " << x << "=/=" << y << std::endl;
        return false;
      }
    }
    return true;
  };

  // Allocating buffers
  const size_t buf_size = grid.size() * sizeof(float);
  cl::Buffer grid_dev =
      OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
                    static_cast<void *>(grid.data()));
  cl::Buffer output_dev = OpenCL::alloc(CL_MEM_READ_WRITE, N * sizeof(float));

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
          r->getKernel().setArg(0, grid_dev);
          r->getKernel().setArg(1, output_dev);
          OpenCL::executeRun<float>(*r, output_dev, N, validate);
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
        r->getKernel().setArg(0, grid_dev);
        r->getKernel().setArg(1, output_dev);
        OpenCL::executeRun<float>(*r, output_dev, N, validate);
      }
    }
  }
};

#endif // EXECUTOR_STENCIL_HARNESS_H
