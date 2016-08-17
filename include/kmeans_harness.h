//
// Created by s1042579 on 17/08/16.
//

#ifndef EXECUTOR_KMEANS_HARNESS_H
#define EXECUTOR_KMEANS_HARNESS_H
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <limits>

#include "run.h"
#include "utils.h"
#include "opencl_utils.h"

/**
 * FIXME: This is a lazy copy paste of the old main with a template switch for single and double precision
 */
template<typename T>
void run_harness(
    std::vector<std::shared_ptr<Run>>& all_run,
    const size_t num_points,
    const size_t num_clusters,
    const size_t num_features,
    const std::string& points_file,
    const std::string& clusters_file,
    const std::string& gold_file,
    const bool force,
    const bool threaded,
    const bool binary,
    const bool transposePoints
)
{
  using namespace std;

  // Compute input and output
  vector<T> points(num_points * num_features, 0);
  vector<T> clusters(num_clusters * num_features, 0);
  vector<int> gold(num_points, 0);

  if (File::is_file_exist(gold_file) &&
      File::is_file_exist(points_file) &&
      File::is_file_exist(clusters_file) &&
      !force) {

    std::cout << "Read data from file..." << std::endl;

    File::load_input(points, points_file);
    File::load_input(clusters, clusters_file);
    File::load_input(gold, gold_file);

  } else {

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0.0,10.0);

    std::cout << "Initialising input data..." << std::endl;

    for (auto &item : points)
      item = distribution(generator);

    for (auto &item : clusters)
      item = distribution(generator);

    std::cout << "Computing gold..." << std::endl;

    // Rodinia 3.1
    for (auto point_id = 0; point_id < num_points; point_id++) {
      {
        int index = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (int i = 0; i < num_clusters; i++) {

          float dist = 0;
          float ans = 0;
          for (int l = 0; l < num_features; l++) {
            ans += (points[l + num_features * point_id] - clusters[i * num_features + l]) *
                   (points[l + num_features * point_id] - clusters[i * num_features + l]);
          }

          dist = ans;
          if (dist < min_dist) {
            min_dist = dist;
            index = i;

          }
        }
        gold[point_id] = index;
      }
    }

    if (transposePoints)
      transpose(points, num_points, num_features);

    std::cout << "Saving to file..." << std::endl;

    File::save_input(points, points_file);
    File::save_input(clusters, clusters_file);
    File::save_input(gold, gold_file);
  }

  // Validation function
  auto validate = [&](const std::vector<int> &output) {
    if(gold.size() != output.size()) return false;
    for(unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      auto y = output[i];

      if (y != x)
        return false;
    }

    return true;
  };

  // Allocating buffers
  const size_t buf_size_points = points.size() * sizeof(T);
  const size_t buf_size_clusters = clusters.size() * sizeof(T);
  const size_t buf_size_output = gold.size() * sizeof(int);

  cl::Buffer points_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      buf_size_points, static_cast<void*>(points.data()) );
  cl::Buffer clusters_dev = OpenCL::alloc( CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      buf_size_clusters, static_cast<void*>(clusters.data()));

  cl::Buffer output_dev = OpenCL::alloc( CL_MEM_READ_WRITE, buf_size_output );

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
          while (!ready) cv.wait(locker);
        }

        while (!ready_queue.empty()) {
          {
            std::unique_lock<std::mutex> locker(m);
            r = ready_queue.front();
            ready_queue.pop();
          }
          r->getKernel().setArg(0, points_dev);
          r->getKernel().setArg(1, clusters_dev);
          r->getKernel().setArg(2, output_dev);
          OpenCL::executeRun<int>(*r, output_dev, num_points, validate);
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
        r->getKernel().setArg(0, points_dev);
        r->getKernel().setArg(1, clusters_dev);
        r->getKernel().setArg(2, output_dev);
        OpenCL::executeRun<int>(*r, output_dev, num_points, validate);
      }
    }
  }
};

#endif //EXECUTOR_KMEANS_HARNESS_H
