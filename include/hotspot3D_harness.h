#ifndef EXECUTOR_HOTSPOT3D_HARNESS_H
#define EXECUTOR_HOTSPOT3D_HARNESS_H

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

void set_kernel_args(const shared_ptr<Run> run, const cl::Buffer &temp_dev,
                     const cl::Buffer &power_dev, const cl::Buffer &output_dev,
                     const size_t M, const size_t N, const size_t O) {
  unsigned i = 0;
  float ce = 8.3333325E-6;
  float cw = 8.3333325E-6;
  float cn = 8.3333325E-6;
  float cs = 8.3333325E-6;
  float ct = 2.6666664E-4;
  float cb = 2.6666664E-4;
  float cc = 0.99916667;
  float stepDivCap = 8.3333325E-5;
  run->getKernel().setArg(i++, temp_dev);
  run->getKernel().setArg(i++, power_dev);
  run->getKernel().setArg(i++, ce);
  run->getKernel().setArg(i++, cw);
  run->getKernel().setArg(i++, cn);
  run->getKernel().setArg(i++, cs);
  run->getKernel().setArg(i++, ct);
  run->getKernel().setArg(i++, cb);
  run->getKernel().setArg(i++, cc);
  run->getKernel().setArg(i++, stepDivCap);
  run->getKernel().setArg(i++, output_dev);
  // provide extra local memory args if existing
  for (const auto &local_arg : run->extra_local_args)
    run->getKernel().setArg(i++, local_arg);
  run->getKernel().setArg(i++, static_cast<int>(M));
  run->getKernel().setArg(i++, static_cast<int>(N));
  run->getKernel().setArg(i++, static_cast<int>(O));
}

float calculateHotspot(float tInC, float cc, float tInN, float cn, float tInS,
                       float cs, float tInE, float ce, float tInW, float cw,
                       float tInT, float ct, float tInB, float cb,
                       float stepDivCap, float pInC, float amb_temp) {
  return tInC * cc + tInN * cn + tInS * cs + tInE * ce + tInW * cw + tInT * ct +
         tInB * cb + stepDivCap * pInC + ct * amb_temp;
}

float id(float x) { return x; }

void compute_gold(const size_t M, const size_t N, const size_t O,
                  Matrix<float> &temp, Matrix<float> &power,
                  Matrix<float> &gold, const std::string &temp_file,
                  const std::string &power_file, const std::string &gold_file) {

  File::load_input_debug(
      temp,
      "/home/bastian/development/exploration/datasets/hotspot3D/temp.txt");
  File::load_input_debug(
      power,
      "/home/bastian/development/exploration/datasets/hotspot3D/power.txt");
  File::load_input_debug(
      gold,
      "/home/bastian/development/exploration/datasets/hotspot3D/hotspot3D.txt");

  int v_O_5 = O; // 8
  int v_N_4 = N; // 512
  int v_M_3 = M; // 512
  float v__60 = 8.3333325E-6;
  float v__61 = 8.3333325E-6;
  float v__62 = 8.3333325E-6;
  float v__63 = 8.3333325E-6;
  float v__64 = 2.6666664E-4;
  float v__65 = 2.6666664E-4;
  float v__66 = 0.99916667;
  float v__67 = 8.3333325E-5;

  float v__72;
  float v__74;
  for (int v_gl_id_55 = 0; v_gl_id_55 < v_O_5; v_gl_id_55++) {
    for (int v_gl_id_56 = 0; v_gl_id_56 < v_N_4; v_gl_id_56++) {
      for (int v_gl_id_57 = 0; v_gl_id_57 < v_M_3; v_gl_id_57++) {
        float v_tmp_84 = 80.0f;
        v__72 = v_tmp_84;
        v__74 = calculateHotspot(
            temp[((v_M_3 * v_N_4 *
                   ((v_gl_id_55 >= 0)
                        ? ((v_gl_id_55 < v_O_5)
                               ? v_gl_id_55
                               : (-1 + (2 * v_O_5) + (-1 * v_gl_id_55)))
                        : (-1 + (-1 * v_gl_id_55)))) +
                  (v_M_3 * ((v_gl_id_56 >= 0)
                                ? ((v_gl_id_56 < v_N_4)
                                       ? v_gl_id_56
                                       : (-1 + (2 * v_N_4) + (-1 * v_gl_id_56)))
                                : (-1 + (-1 * v_gl_id_56)))) +
                  ((v_gl_id_57 >= 0)
                       ? ((v_gl_id_57 < v_M_3)
                              ? v_gl_id_57
                              : (-1 + (2 * v_M_3) + (-1 * v_gl_id_57)))
                       : (-1 + (-1 * v_gl_id_57))))],
            v__66,
            temp[((v_M_3 * v_N_4 *
                   ((v_gl_id_55 >= 0)
                        ? ((v_gl_id_55 < v_O_5)
                               ? v_gl_id_55
                               : (-1 + (2 * v_O_5) + (-1 * v_gl_id_55)))
                        : (-1 + (-1 * v_gl_id_55)))) +
                  (v_M_3 * (((-1 + v_gl_id_56) >= 0)
                                ? (((-1 + v_gl_id_56) < v_N_4)
                                       ? (-1 + v_gl_id_56)
                                       : ((-1 * v_gl_id_56) + (2 * v_N_4)))
                                : (-1 * v_gl_id_56))) +
                  ((v_gl_id_57 >= 0)
                       ? ((v_gl_id_57 < v_M_3)
                              ? v_gl_id_57
                              : (-1 + (2 * v_M_3) + (-1 * v_gl_id_57)))
                       : (-1 + (-1 * v_gl_id_57))))],
            v__62,
            temp[((v_M_3 * v_N_4 *
                   ((v_gl_id_55 >= 0)
                        ? ((v_gl_id_55 < v_O_5)
                               ? v_gl_id_55
                               : (-1 + (2 * v_O_5) + (-1 * v_gl_id_55)))
                        : (-1 + (-1 * v_gl_id_55)))) +
                  (v_M_3 * (((1 + v_gl_id_56) >= 0)
                                ? (((1 + v_gl_id_56) < v_N_4)
                                       ? (1 + v_gl_id_56)
                                       : (-2 + (2 * v_N_4) + (-1 * v_gl_id_56)))
                                : (-2 + (-1 * v_gl_id_56)))) +
                  ((v_gl_id_57 >= 0)
                       ? ((v_gl_id_57 < v_M_3)
                              ? v_gl_id_57
                              : (-1 + (2 * v_M_3) + (-1 * v_gl_id_57)))
                       : (-1 + (-1 * v_gl_id_57))))],
            v__63,
            temp[((v_M_3 * v_N_4 *
                   (((1 + v_gl_id_55) >= 0)
                        ? (((1 + v_gl_id_55) < v_O_5)
                               ? (1 + v_gl_id_55)
                               : (-2 + (2 * v_O_5) + (-1 * v_gl_id_55)))
                        : (-2 + (-1 * v_gl_id_55)))) +
                  (v_M_3 * ((v_gl_id_56 >= 0)
                                ? ((v_gl_id_56 < v_N_4)
                                       ? v_gl_id_56
                                       : (-1 + (2 * v_N_4) + (-1 * v_gl_id_56)))
                                : (-1 + (-1 * v_gl_id_56)))) +
                  ((v_gl_id_57 >= 0)
                       ? ((v_gl_id_57 < v_M_3)
                              ? v_gl_id_57
                              : (-1 + (2 * v_M_3) + (-1 * v_gl_id_57)))
                       : (-1 + (-1 * v_gl_id_57))))],
            v__60,
            temp[((v_M_3 * v_N_4 *
                   (((-1 + v_gl_id_55) >= 0)
                        ? (((-1 + v_gl_id_55) < v_O_5)
                               ? (-1 + v_gl_id_55)
                               : ((-1 * v_gl_id_55) + (2 * v_O_5)))
                        : (-1 * v_gl_id_55))) +
                  (v_M_3 * ((v_gl_id_56 >= 0)
                                ? ((v_gl_id_56 < v_N_4)
                                       ? v_gl_id_56
                                       : (-1 + (2 * v_N_4) + (-1 * v_gl_id_56)))
                                : (-1 + (-1 * v_gl_id_56)))) +
                  ((v_gl_id_57 >= 0)
                       ? ((v_gl_id_57 < v_M_3)
                              ? v_gl_id_57
                              : (-1 + (2 * v_M_3) + (-1 * v_gl_id_57)))
                       : (-1 + (-1 * v_gl_id_57))))],
            v__61,
            temp[((v_M_3 * v_N_4 *
                   ((v_gl_id_55 >= 0)
                        ? ((v_gl_id_55 < v_O_5)
                               ? v_gl_id_55
                               : (-1 + (2 * v_O_5) + (-1 * v_gl_id_55)))
                        : (-1 + (-1 * v_gl_id_55)))) +
                  (v_M_3 * ((v_gl_id_56 >= 0)
                                ? ((v_gl_id_56 < v_N_4)
                                       ? v_gl_id_56
                                       : (-1 + (2 * v_N_4) + (-1 * v_gl_id_56)))
                                : (-1 + (-1 * v_gl_id_56)))) +
                  (((1 + v_gl_id_57) >= 0)
                       ? (((1 + v_gl_id_57) < v_M_3)
                              ? (1 + v_gl_id_57)
                              : (-2 + (2 * v_M_3) + (-1 * v_gl_id_57)))
                       : (-2 + (-1 * v_gl_id_57))))],
            v__64,
            temp[((v_M_3 * v_N_4 *
                   ((v_gl_id_55 >= 0)
                        ? ((v_gl_id_55 < v_O_5)
                               ? v_gl_id_55
                               : (-1 + (2 * v_O_5) + (-1 * v_gl_id_55)))
                        : (-1 + (-1 * v_gl_id_55)))) +
                  (v_M_3 * ((v_gl_id_56 >= 0)
                                ? ((v_gl_id_56 < v_N_4)
                                       ? v_gl_id_56
                                       : (-1 + (2 * v_N_4) + (-1 * v_gl_id_56)))
                                : (-1 + (-1 * v_gl_id_56)))) +
                  (((-1 + v_gl_id_57) >= 0)
                       ? (((-1 + v_gl_id_57) < v_M_3)
                              ? (-1 + v_gl_id_57)
                              : ((-1 * v_gl_id_57) + (2 * v_M_3)))
                       : (-1 * v_gl_id_57)))],
            v__65, v__67, power[(v_gl_id_57 + (v_M_3 * v_N_4 * v_gl_id_55) +
                                 (v_M_3 * v_gl_id_56))],
            v__72);
        gold[(v_gl_id_57 + (v_M_3 * v_N_4 * v_gl_id_55) +
              (v_M_3 * v_gl_id_56))] = id(v__74);
      }
    }
  }

  File::save_input(gold, gold_file);
  File::save_input(temp, temp_file);
  File::save_input(power, power_file);
}

void run_harness(std::vector<std::shared_ptr<Run>> &all_run, const size_t M,
                 const size_t N, const size_t O, const std::string &temp_file,
                 const std::string &power_file, const std::string &gold_file,
                 const bool force, const bool threaded, const bool binary) {

  if (binary)
    std::cout << "Using precompiled binaries" << std::endl;

  // M rows, N columns
  Matrix<float> temp(M * N * O);
  Matrix<float> power(M * N * O);
  Matrix<float> gold(M * N * O);

  // use existing grid, weights and gold or init them
  if (File::is_file_exist(gold_file) && File::is_file_exist(temp_file) &&
      File::is_file_exist(power_file)) {
    std::cout << "use existing grid, weights and gold" << std::endl;
    File::load_input(gold, gold_file);
    File::load_input(temp, temp_file);
    File::load_input(power, power_file);
  } else {
    std::cout << "load files and save as binary" << std::endl;
    compute_gold(M, N, O, temp, power, gold, temp_file, power_file, gold_file);
  }

  // validation function
  auto validate = [&](const std::vector<float> &output) {
    bool correct = true;
    if (gold.size() != output.size())
      return false;
    for (unsigned i = 0; i < gold.size(); ++i) {
      auto x = gold[i];
      // auto x = temp[i];
      auto y = output[i];

      // possibly lots of floating point weirdness going on
      if (abs(x - y) > 0.001f * max(abs(x), abs(y))) {
        cerr << "at " << i << ": " << x << "=/=" << y << std::endl;
        return false;
        // correct = false;
      }
    }
    return correct;
  };

  // Allocating buffers
  const size_t buf_size = temp.size() * sizeof(float);
  const size_t out_size = gold.size() * sizeof(float);
  cl::Buffer temp_dev =
      OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
                    static_cast<void *>(temp.data()));
  cl::Buffer power_dev =
      OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
                    static_cast<void *>(power.data()));
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

          set_kernel_args(r, temp_dev, power_dev, output_dev, M, N, O);
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
        set_kernel_args(r, temp_dev, power_dev, output_dev, M, N, O);
        OpenCL::executeRun<float>(*r, output_dev, gold.size(), validate);
      }
    }
  }
};

#endif // EXECUTOR_HOTSPOT3D_HARNESS_H
