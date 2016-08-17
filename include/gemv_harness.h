//
// Created by s1042579 on 17/08/16.
//

#ifndef EXECUTOR_GEMV_HARNESS_H
#define EXECUTOR_GEMV_HARNESS_H

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "run.h"
#include "utils.h"
#include "opencl_utils.h"

/**
 * FIXME: This is a lazy copy paste of the old main with a template switch for single and double precision
 */
void run_harness(
    std::vector<std::shared_ptr<Run>>& all_run,
    const size_t N,
    const std::string& mat_file,
    const std::string& vecX_file,
    const std::string& vecY_file,
    const std::string& gold_file,
    const bool force,
    const bool transposeIn,
    const bool threaded,
    const bool binary
)
{
  using namespace std;

  float alpha = 1.5;
  float beta = 2.5;

  if(binary)
    std::cout << "Using precompiled binaries" << std::endl;
  // Compute input and output
  Matrix<float> mat(N*N);
  std::vector<float> vecX(N);
  std::vector<float> vecY(N);
  std::vector<float> gold(N);

  if(File::is_file_exist(gold_file) && File::is_file_exist(mat_file) &&
      File::is_file_exist(vecX_file) && File::is_file_exist(vecY_file) && !force ) {
    File::load_input(gold, gold_file);
    File::load_input(mat, mat_file);
    File::load_input(vecX, vecX_file);
    File::load_input(vecY, vecY_file);
  } else {
    for(unsigned y = 0; y < N; ++y) {
      for (unsigned x = 0; x < N; ++x) {
        mat[y * N + x] = (((y * 3 + x * 2) % 10) + 1) * 1.0f;
      }
      vecX[y] = (y%10)*0.5f;
      vecY[y] = (y%10)*1.5f;
    }

    // compute gold
    for (unsigned i=0; i<N; i++) {
      float Result=0.0;
      for (unsigned j=0; j<N; j++)
        Result+=mat[i*N+j]*vecX[j];

      gold[i]=Result * alpha + vecY[i] * beta;
    }

    if (transposeIn) {
      std::vector<float> Tmat(N*N);
      for(unsigned y = 0; y < N; ++y)
        for(unsigned x = 0; x < N; ++x)
          Tmat[y*N+x] = mat[x*N+y];
      std::swap(Tmat, mat);
    }

    File::save_input(gold, gold_file);
    File::save_input(mat, mat_file);
    File::save_input(vecX, vecX_file);
    File::save_input(vecY, vecY_file);
  }

  // validation function
  auto validate = [&](const std::vector<float> &output) {
    if(gold.size() != output.size()) return false;
    for(unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      auto y = output[i];

      if(abs(x - y) > 0.001f * max(abs(x), abs(y))) {
        cout << "at " << i << ": " << x << "=/=" << y <<std::endl;
        return false;
      }
    }
    return true;
  };

  // Allocating buffers
  const size_t buf_size = mat.size() * sizeof(float);
  cl::Buffer mat_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      buf_size, static_cast<void*>(mat.data()) );
  cl::Buffer vecX_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       N*sizeof(float), static_cast<void*>(vecX.data()) );
  cl::Buffer vecY_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       N*sizeof(float), static_cast<void*>(vecY.data()) );
  cl::Buffer output_dev = OpenCL::alloc( CL_MEM_READ_WRITE, N*sizeof(float) );

  // multi-threaded exec
  if(threaded) {
    std::mutex m;
    std::condition_variable cv;

    bool done = false;
    bool ready = false;
    std::queue<std::shared_ptr<Run>> ready_queue;

    // compilation thread
    auto compilation_thread = std::thread([&] {
      for (auto &r: all_run) {
        if (r->compile(binary)) {
          std::unique_lock<std::mutex> locker(m);
          ready_queue.push(r);
          ready = true;
          cv.notify_one();
        }
      }
    });

    auto execute_thread = std::thread([&] {
      std::shared_ptr<Run> r = nullptr;
      while (!done) {
        {
          std::unique_lock<std::mutex> locker(m);
          while (!ready && !done) cv.wait(locker);
        }

        while (!ready_queue.empty()) {
          {
            std::unique_lock<std::mutex> locker(m);
            r = ready_queue.front();
            ready_queue.pop();
          }
          r->getKernel().setArg(0, mat_dev);
          r->getKernel().setArg(1, vecX_dev);
          r->getKernel().setArg(2, vecY_dev);
          r->getKernel().setArg(3, alpha);
          r->getKernel().setArg(4, beta);
          r->getKernel().setArg(5, output_dev);
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
    for (auto &r: all_run) {
      if (r->compile(binary)) {
        r->getKernel().setArg(0, mat_dev);
        r->getKernel().setArg(1, vecX_dev);
        r->getKernel().setArg(2, vecY_dev);
        r->getKernel().setArg(3, alpha);
        r->getKernel().setArg(4, beta);
        r->getKernel().setArg(5, output_dev);
        OpenCL::executeRun<float>(*r, output_dev, N, validate);
      }
    }
  }
};

#endif //EXECUTOR_GEMV_HARNESS_H
