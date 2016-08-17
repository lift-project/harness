//
// Created by s1042579 on 17/08/16.
//

#ifndef EXECUTOR_NBODY_HARNESS_H
#define EXECUTOR_NBODY_HARNESS_H

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
template<typename T>
void run_harness(
    std::vector<std::shared_ptr<Run>>& all_run,
    const size_t size,
    float espSqr,
    float deltaT,
    const std::string& positions_file,
    const std::string& velocities_file,
    const std::string& gold_file,
    const bool force,
    const bool threaded,
    const bool binary
)
{
  using namespace std;

  // Compute input and output
  vector<T> positions(4 * size, 0);
  vector<T> velocities(4 * size, 0);
  vector<T> gold(8 * size, 0);

  if (File::is_file_exist(gold_file) &&
      File::is_file_exist(positions_file) &&
      File::is_file_exist(velocities_file) &&
      !force) {

    std::cout << "Read data from file..." << std::endl;

    File::load_input(positions, positions_file);
    File::load_input(velocities, velocities_file);
    File::load_input(gold, gold_file);

  } else {

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0.0,10.0);

    std::cout << "Initialising input data..." << std::endl;

    for (auto &item : positions)
      item = distribution(generator);

    for (auto i = 0; i < size; i++)
      velocities[4*i+3] = positions[4*i+3];

    std::cout << "Computing gold..." << std::endl;

    // AMD SDK
    // Iterate for all samples
    for(cl_uint i = 0; i < size; ++i)
    {
      int myIndex = 4 * i;
      float acc[3] = {0.0f, 0.0f, 0.0f};
      for(cl_uint j = 0; j < size; ++j)
      {
        float r[3];
        int index = 4 * j;

        float distSqr = 0.0f;
        for(int k = 0; k < 3; ++k)
        {
          r[k] = positions[index + k] - positions[myIndex + k];

          distSqr += r[k] * r[k];
        }

        float invDist = 1.0f / sqrt(distSqr + espSqr);
        float invDistCube =  invDist * invDist * invDist;
        float s = positions[index + 3] * invDistCube;

        for(int k = 0; k < 3; ++k)
        {
          acc[k] += s * r[k];
        }
      }

      for(int k = 0; k < 3; ++k)
      {
        gold[myIndex*2 + k] = positions[myIndex + k] + velocities[myIndex + k] * deltaT +
                              0.5f * acc[k] * deltaT * deltaT;
        gold[myIndex*2 + k + 4] = velocities[myIndex + k] + acc[k] * deltaT;
      }
      gold[myIndex*2+3] = positions[myIndex + 3];
      gold[myIndex*2+3+4] = velocities[myIndex + 3];
    }

    std::cout << "Saving to file..." << std::endl;

    File::save_input(positions, positions_file);
    File::save_input(velocities, velocities_file);
    File::save_input(gold, gold_file);
  }

  // Validation function
  auto validate = [&](const std::vector<T> &output) {
    if(gold.size() != output.size()) return false;
    for(unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      auto y = output[i];

      if(abs(x - y) > 0.001f * max(abs(x), abs(y)))
        return false;
    }

    return true;
  };

  // Allocating buffers
  const size_t buf_size_positions = positions.size() * sizeof(T);
  const size_t buf_size_velocities = velocities.size() * sizeof(T);
  const size_t buf_size_output = gold.size() * sizeof(T);

  cl::Buffer positions_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      buf_size_positions, static_cast<void*>(positions.data()) );
  cl::Buffer velocities_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      buf_size_velocities, static_cast<void*>(velocities.data()));

  cl::Buffer output_dev = OpenCL::alloc( CL_MEM_READ_WRITE, buf_size_output );

  // multi-threaded exec
  if(threaded) {
    std::mutex m;
    std::condition_variable cv;

    bool done = false;
    bool ready = false;
    std::queue<Run*> ready_queue;

    // compilation thread
    auto compilation_thread = std::thread([&] {
      for (auto &r: all_run) {
        if (r->compile(binary)) {
          std::unique_lock<std::mutex> locker(m);
          ready_queue.push(&*r);
          ready = true;
          cv.notify_one();
        }
      }
    });

    auto execute_thread = std::thread([&] {
      Run *r = nullptr;
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
          r->getKernel().setArg(0, positions_dev);
          r->getKernel().setArg(1, velocities_dev);
          r->getKernel().setArg(2, espSqr);
          r->getKernel().setArg(3, deltaT);
          r->getKernel().setArg(4, output_dev);
          OpenCL::executeRun<T>(*r, output_dev, size*8, validate);
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
      if (r->compile(binary)) {
        r->getKernel().setArg(0, positions_dev);
        r->getKernel().setArg(1, velocities_dev);
        r->getKernel().setArg(2, espSqr);
        r->getKernel().setArg(3, deltaT);
        r->getKernel().setArg(4, output_dev);
        OpenCL::executeRun<T>(*r, output_dev, size*8, validate);
      }
    }
  }
};

#endif //EXECUTOR_NBODY_HARNESS_H
