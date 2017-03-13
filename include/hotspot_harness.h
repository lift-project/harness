//
// Created by Bastian Hagedorn
//

#ifndef EXECUTOR_HOTSPOT_HARNESS_H
#define EXECUTOR_HOTSPOT_HARNESS_H

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
		     const cl::Buffer &power_dev, const cl::Buffer &output_dev, const size_t M,
		     const size_t N) {
	unsigned i = 0;
	run->getKernel().setArg(i++, temp_dev);
	run->getKernel().setArg(i++, power_dev);
	run->getKernel().setArg(i++, output_dev);
	// provide extra local memory args if existing
	for (const auto &local_arg : run->extra_local_args)
		run->getKernel().setArg(i++, local_arg);
	run->getKernel().setArg(i++, static_cast<int>(M));
	run->getKernel().setArg(i++, static_cast<int>(N));
}

float rodiniaUserFun(float power, float top, float bottom, float left, float right, float center) {
	{
		float step_div_cap = 1.365333e+00;
		return center +
		       step_div_cap *
			   (power + 0.1f * (bottom + top - 2 * center) +
			    0.1 * (left + right - 2 * center) + 4.882813e-05 * (80.0f - center));
	}
}

float id(float x) {
	{ return x; }
}

void compute_gold(const size_t M, const size_t N, Matrix<float> &temp, Matrix<float> &power,
		  Matrix<float> &gold, const std::string &temp_file, const std::string &power_file,
		  const std::string &gold_file) {

	// File::load_input_debug(gold, "/home/bastian/development/exploration/executor/datasets/"
	//			     "rodiniaData/lift_hotspot_gold_8192_nonBinary");
	File::load_input_debug(temp, "/home/bastian/development/exploration/executor/datasets/"
				     "rodiniaData/lift_hotspot_temp_8192_nonBinary");
	File::load_input_debug(power, "/home/bastian/development/exploration/executor/datasets/"
				      "rodiniaData/lift_hotspot_power_8192_nonBinary");
	/*
	// taken from generated kernel
	//
	float v__5;
	int v_M_3 = M;
	int v_N_4 = N;
	for (int v_gl_id_6 = 0; v_gl_id_6 < 8192; v_gl_id_6++) {
		for (int v_gl_id_7 = 0; v_gl_id_7 < 8192; v_gl_id_7++) {
			v__5 = rodiniaUserFun(
			    power[(v_gl_id_7 + (v_M_3 * v_gl_id_6))],
			    temp[((v_M_3 * (((-1 + v_gl_id_6 + (v_gl_id_7 / v_M_3)) >= 0)
						? (((-1 + v_gl_id_6 + (v_gl_id_7 / v_M_3)) < v_N_4)
						       ? (-1 + v_gl_id_6 + (v_gl_id_7 / v_M_3))
						       : (-1 + v_N_4))
						: 0)) +
				  (((v_gl_id_7 % v_M_3) >= 0)
				       ? (((v_gl_id_7 % v_M_3) < v_M_3) ? (v_gl_id_7 % v_M_3)
									: (-1 + v_M_3))
				       : 0))],
			    temp[((v_M_3 * (((1 + v_gl_id_6 + (v_gl_id_7 / v_M_3)) >= 0)
						? (((1 + v_gl_id_6 + (v_gl_id_7 / v_M_3)) < v_N_4)
						       ? (1 + v_gl_id_6 + (v_gl_id_7 / v_M_3))
						       : (-1 + v_N_4))
						: 0)) +
				  (((v_gl_id_7 % v_M_3) >= 0)
				       ? (((v_gl_id_7 % v_M_3) < v_M_3) ? (v_gl_id_7 % v_M_3)
									: (-1 + v_M_3))
				       : 0))],
			    temp[((v_M_3 * (((v_gl_id_6 + (v_gl_id_7 / v_M_3)) >= 0)
						? (((v_gl_id_6 + (v_gl_id_7 / v_M_3)) < v_N_4)
						       ? (v_gl_id_6 + (v_gl_id_7 / v_M_3))
						       : (-1 + v_N_4))
						: 0)) +
				  (((-1 + (v_gl_id_7 % v_M_3)) >= 0)
				       ? (((-1 + (v_gl_id_7 % v_M_3)) < v_M_3)
					      ? (-1 + (v_gl_id_7 % v_M_3))
					      : (-1 + v_M_3))
				       : 0))],
			    temp[((v_M_3 * (((v_gl_id_6 + (v_gl_id_7 / v_M_3)) >= 0)
						? (((v_gl_id_6 + (v_gl_id_7 / v_M_3)) < v_N_4)
						       ? (v_gl_id_6 + (v_gl_id_7 / v_M_3))
						       : (-1 + v_N_4))
						: 0)) +
				  (((1 + (v_gl_id_7 % v_M_3)) >= 0)
				       ? (((1 + (v_gl_id_7 % v_M_3)) < v_M_3)
					      ? (1 + (v_gl_id_7 % v_M_3))
					      : (-1 + v_M_3))
				       : 0))],
			    temp[((v_M_3 * (((v_gl_id_6 + (v_gl_id_7 / v_M_3)) >= 0)
						? (((v_gl_id_6 + (v_gl_id_7 / v_M_3)) < v_N_4)
						       ? (v_gl_id_6 + (v_gl_id_7 / v_M_3))
						       : (-1 + v_N_4))
						: 0)) +
				  (((v_gl_id_7 % v_M_3) >= 0)
				       ? (((v_gl_id_7 % v_M_3) < v_M_3) ? (v_gl_id_7 % v_M_3)
									: (-1 + v_M_3))
				       : 0))]);
			gold[(v_gl_id_7 + (v_M_3 * v_gl_id_6))] = id(v__5);
		}
	}
	*/

	//  correct version - taken from passing lift test
	int v_M_0 = M;
	int v_N_1 = N;
	float v__19;
	for (int v_gl_id_11 = 0; v_gl_id_11 < 8192; v_gl_id_11++) {
		for (int v_gl_id_12 = 0; v_gl_id_12 < 8192; v_gl_id_12++) {
			v__19 = rodiniaUserFun(
			    power[(v_gl_id_12 + (v_M_0 * v_gl_id_11))],
			    temp[((v_M_0 * (((-1 + v_gl_id_11) >= 0)
						? (((-1 + v_gl_id_11) < v_N_1) ? (-1 + v_gl_id_11)
									       : (-1 + v_N_1))
						: 0)) +
				  ((v_gl_id_12 >= 0)
				       ? ((v_gl_id_12 < v_M_0) ? v_gl_id_12 : (-1 + v_M_0))
				       : 0))],
			    temp[((v_M_0 * (((1 + v_gl_id_11) >= 0)
						? (((1 + v_gl_id_11) < v_N_1) ? (1 + v_gl_id_11)
									      : (-1 + v_N_1))
						: 0)) +
				  ((v_gl_id_12 >= 0)
				       ? ((v_gl_id_12 < v_M_0) ? v_gl_id_12 : (-1 + v_M_0))
				       : 0))],
			    temp[((v_M_0 * ((v_gl_id_11 >= 0)
						? ((v_gl_id_11 < v_N_1) ? v_gl_id_11 : (-1 + v_N_1))
						: 0)) +
				  (((-1 + v_gl_id_12) >= 0)
				       ? (((-1 + v_gl_id_12) < v_M_0) ? (-1 + v_gl_id_12)
								      : (-1 + v_M_0))
				       : 0))],
			    temp[((v_M_0 * ((v_gl_id_11 >= 0)
						? ((v_gl_id_11 < v_N_1) ? v_gl_id_11 : (-1 + v_N_1))
						: 0)) +
				  (((1 + v_gl_id_12) >= 0)
				       ? (((1 + v_gl_id_12) < v_M_0) ? (1 + v_gl_id_12)
								     : (-1 + v_M_0))
				       : 0))],
			    temp[((v_M_0 * ((v_gl_id_11 >= 0)
						? ((v_gl_id_11 < v_N_1) ? v_gl_id_11 : (-1 + v_N_1))
						: 0)) +
				  ((v_gl_id_12 >= 0)
				       ? ((v_gl_id_12 < v_M_0) ? v_gl_id_12 : (-1 + v_M_0))
				       : 0))]);
			gold[(v_gl_id_12 + (v_M_0 * v_gl_id_11))] = id(v__19);
		}
	}

	File::save_input(gold, gold_file);
	File::save_input(temp, temp_file);
	File::save_input(power, power_file);
}

void run_harness(std::vector<std::shared_ptr<Run>> &all_run, const size_t M, const size_t N,
		 const std::string &temp_file, const std::string &power_file,
		 const std::string &gold_file, const bool force, const bool threaded,
		 const bool binary) {

	if (binary) std::cout << "Using precompiled binaries" << std::endl;

	// M rows, N columns
	Matrix<float> temp(M * N);
	Matrix<float> power(M * N);
	Matrix<float> gold(M * N);

	// use existing grid, weights and gold or init them
	if (File::is_file_exist(gold_file) && File::is_file_exist(temp_file) &&
	    File::is_file_exist(power_file)) {
		std::cout << "use existing grid, weights and gold" << std::endl;
		File::load_input(gold, gold_file);
		File::load_input(temp, temp_file);
		File::load_input(power, power_file);
	} else {
		std::cout << "load files and save as binary" << std::endl;
		compute_gold(M, N, temp, power, gold, temp_file, power_file, gold_file);
	}

	// validation function
	auto validate = [&](const std::vector<float> &output) {
		bool correct = true;
		if (gold.size() != output.size()) return false;
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
	cl::Buffer temp_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					    static_cast<void *>(temp.data()));
	cl::Buffer power_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
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

					set_kernel_args(r, temp_dev, power_dev, output_dev, M, N);
					OpenCL::executeRun<float>(*r, output_dev, gold.size(),
								  validate);
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
				set_kernel_args(r, temp_dev, power_dev, output_dev, M, N);
				OpenCL::executeRun<float>(*r, output_dev, gold.size(), validate);
			}
		}
	}
};

#endif // EXECUTOR_HOTSPOT_HARNESS_H
