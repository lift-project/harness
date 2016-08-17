//
// Created by s1042579 on 16/08/16.
//

#ifndef EXECUTOR_MM_HARNESS_H
#define EXECUTOR_MM_HARNESS_H

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <CL/cl.hpp>

#include "run.h"
#include "utils.h"
#include "opencl_utils.h"

template <typename T>
void execute_run(
    const size_t &M,
    const size_t &N,
    std::function<bool(const std::vector<T>&)> validate,
    const cl::Buffer &matA_dev,
    const cl::Buffer &matB_dev,
    const cl::Buffer &output_dev,
    std::shared_ptr<Run> &r
) {
  r->getKernel().setArg(0, matA_dev);
  r->getKernel().setArg(1, matB_dev);
  r->getKernel().setArg(2, output_dev);
  OpenCL::executeRun<T>(*r, output_dev, N * M, validate);
};


/**
 * FIXME: This is a lazy copy paste of the old main with a template switch for single and double precision
 */
template<typename T>
void run_harness(
    std::vector<std::shared_ptr<Run>>& all_run,
    const size_t M,
    const size_t K,
    const size_t N,
    const std::string& matA_file,
    const std::string& matB_file,
    const std::string& gold_file,
    const bool force,
    const bool transposeA,
    const bool transposeB,
    const bool transposeOut,
    const bool threaded,
    const bool binary
)
{
  using namespace std;

  // Compute input and output
  Matrix<T> matA(M * K);
  Matrix<T> matB(K * N);
  Matrix<T> gold(M * N);


  if (File::is_file_exist(gold_file) &&
      File::is_file_exist(matA_file) &&
      File::is_file_exist(matB_file) &&
      !force ) {

    std::cout << "Read matrices from file..." << std::endl;

    File::load_input(gold, gold_file);
    File::load_input(matA, matA_file);
    File::load_input(matB, matB_file);

  } else {

    std::cout << "Initialising matrices..." << std::endl;

    for(unsigned y = 0; y < M; ++y)
      for(unsigned x = 0; x < K; ++x)
      {
        matA[y*K+x] = (((y * 3 + x * 2) % 10) + 1) * 1.0f;
      }

    for(unsigned y = 0; y < K; ++y)
      for(unsigned x = 0; x < N; ++x)
      {
        matB[y*N+x] = (((y * 7 + x * 3) % 10) + 1) * 1.0f;
      }

    // compute gold
    std::vector < std::thread > threads;
    auto mmult = [&](size_t from, size_t to) {
      for (auto i = from; i < to; i++) {
        for (unsigned j = 0; j < N; j++)
          gold[i*N + j] = 0;

        for (unsigned j = 0; j < N; j++)
          for (unsigned k = 0; k < K; k++)
            gold[i*N + j] += matA[i*K + k] * matB[k*N + j];
      }
    };

    std::cout << "Computing gold..." << std::endl;

    auto n_threads = std::thread::hardware_concurrency();
    if(M % n_threads != 0)
      n_threads = 16;
    assert(M % n_threads == 0);
    auto chunk = M / n_threads;
    for (unsigned tid = 0; tid < n_threads; tid++)
      threads.push_back(std::thread([=]{mmult(tid*chunk, (tid+1)*chunk);}));
    for (auto & t : threads) t.join();

    std::cout << "Transposing matrices..." << std::endl;

    if (transposeA)
      transpose(matA, M, K);

    if (transposeB)
      transpose(matB, K, N);

    if (transposeOut)
      transpose(gold, M, N);

    std::cout << "Saving to file..." << std::endl;

    File::save_input(gold, gold_file);
    File::save_input(matA, matA_file);
    File::save_input(matB, matB_file);
  }

  // validation function
  std::function<bool(const std::vector<T>&)> validate = [&](const std::vector<T> &output) {
    if(gold.size() != output.size())
      return false;
    for(unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      auto y = output[i];

      if(abs(x - y) > 0.0001f * max(abs(x), abs(y)))
        return false;
    }

    return true;
  };

  // Allocating buffers
  const size_t buf_size_A = matA.size() * sizeof(T);
  const size_t buf_size_B = matB.size() * sizeof(T);
  const size_t buf_size_C = gold.size() * sizeof(T);
  cl::Buffer matA_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      buf_size_A, static_cast<void*>(matA.data()) );
  cl::Buffer matB_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      buf_size_B, static_cast<void*>(matB.data()) );
  cl::Buffer output_dev = OpenCL::alloc( CL_MEM_READ_WRITE, buf_size_C );

  // multi-threaded exec
  if(threaded) {
    std::mutex m;
    std::condition_variable cv;

    bool done = false;
    bool ready = false;
    std::queue<shared_ptr<Run>> ready_queue;

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
          while (!ready) cv.wait(locker);
        }

        while (!ready_queue.empty()) {
          {
            std::unique_lock<std::mutex> locker(m);
            r = ready_queue.front();
            ready_queue.pop();
          }
          execute_run(M, N, validate, matA_dev, matB_dev, output_dev, r);
        }
      }
    });

    compilation_thread.join();
    {
      std::unique_lock<std::mutex> locker(m);
      done = true;
      cv.notify_one();
    }
    execute_thread.join();
  }
    // single threaded exec
  else {
    for (auto &r: all_run) {
      if (r->compile(binary))
        execute_run(M, N, validate, matA_dev, matB_dev, output_dev, r);
    }
  }
}

#endif //EXECUTOR_MM_HARNESS_H
