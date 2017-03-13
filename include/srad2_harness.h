#ifndef EXECUTOR_SRAD2_HARNESS_H
#define EXECUTOR_SRAD2_HARNESS_H

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

void set_kernel_args(const shared_ptr<Run> run, const cl::Buffer &image_dev,
		     const cl::Buffer &coeffs_dev, const cl::Buffer &dn_dev,const cl::Buffer &ds_dev,const cl::Buffer &de_dev,const cl::Buffer &dw_dev, const cl::Buffer &output_dev, const size_t M,
		     const size_t N) {
	unsigned i = 0;
	run->getKernel().setArg(i++, image_dev);
	run->getKernel().setArg(i++, coeffs_dev);
	run->getKernel().setArg(i++, dn_dev);
	run->getKernel().setArg(i++, ds_dev);
	run->getKernel().setArg(i++, de_dev);
	run->getKernel().setArg(i++, dw_dev);
	run->getKernel().setArg(i++, output_dev);
	// provide extra local memory args if existing
	for (const auto &local_arg : run->extra_local_args)
		run->getKernel().setArg(i++, local_arg);
	run->getKernel().setArg(i++, static_cast<int>(M));
	run->getKernel().setArg(i++, static_cast<int>(N));
}

void compute_gold(const size_t M, const size_t N, Matrix<float> &image, Matrix<float> &coeffs,
                  Matrix<float> &dDN,Matrix<float> &dDS,Matrix<float> &dDE,Matrix<float> &dDW,
		  Matrix<float> &gold, const std::string &image_file, const std::string &coeffs_file,
                  const std::string &dDN_file,const std::string &dDS_file,const std::string &dDE_file,
                  const std::string &dDW_file,
		  const std::string &gold_file) {

        File::load_input_debug(gold,"../datasets/srad_data/finalImage.txt");
	File::load_input_debug(image,"../datasets/srad_data/imagebefore.txt");
	File::load_input_debug(coeffs,"../datasets/srad_data/coeffs.txt");
	File::load_input_debug(dDN,"../datasets/srad_data/ddN.txt");
	File::load_input_debug(dDS,"../datasets/srad_data/dDS.txt");
	File::load_input_debug(dDE,"../datasets/srad_data/dDE.txt");
	File::load_input_debug(dDW,"../datasets/srad_data/dDW.txt");

	File::save_input(gold, gold_file);
	File::save_input(image, image_file);
	File::save_input(coeffs, coeffs_file);
	File::save_input(dDN, dDN_file);
	File::save_input(dDS, dDS_file);
	File::save_input(dDE, dDE_file);
	File::save_input(dDW, dDW_file);
}

void run_harness(std::vector<std::shared_ptr<Run>> &all_run, const size_t M, const size_t N,
		 const std::string &image_file, const std::string &coeffs_file,
                  const std::string &dDN_file,const std::string &dDS_file,const std::string &dDE_file,
                  const std::string &dDW_file,
		 const std::string &gold_file, const bool force, const bool threaded,
		 const bool binary) {

	if (binary) std::cout << "Using precompiled binaries" << std::endl;

	// M rows, N columns
	Matrix<float> image(M * N);
	Matrix<float> coeffs(M * N);
	Matrix<float> dDN(M * N);
	Matrix<float> dDS(M * N);
	Matrix<float> dDE(M * N);
	Matrix<float> dDW(M * N);
	Matrix<float> gold(M * N);

	// use existing grid, weights and gold or init them
	if (File::is_file_exist(gold_file) && File::is_file_exist(image_file) &&
	    File::is_file_exist(coeffs_file) && File::is_file_exist(dDN_file) &&
            File::is_file_exist(dDS_file) && File::is_file_exist(dDE_file) &&
            File::is_file_exist(dDW_file)) {
		std::cout << "use existing grid, weights and gold" << std::endl;
		File::load_input(gold, gold_file);
		File::load_input(image, image_file);
		File::load_input(coeffs, coeffs_file);
		File::load_input(dDN, dDN_file);
		File::load_input(dDS, dDS_file);
		File::load_input(dDE, dDE_file);
		File::load_input(dDW, dDW_file);
	} else {
		std::cout << "load files and save as binary" << std::endl;
		compute_gold(M, N, image, coeffs, dDN, dDS, dDE, dDW, gold, image_file, coeffs_file, dDN_file, dDS_file, dDE_file, dDW_file, gold_file);
	}

	// validation function
	auto validate = [&](const std::vector<float> &output) {
		bool correct = true;
		if (gold.size() != output.size()) return false;
		for (unsigned i = 0; i < gold.size(); ++i) {
			auto x = gold[i];
			// auto x = image[i];
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
	const size_t buf_size = image.size() * sizeof(float);
	const size_t out_size = gold.size() * sizeof(float);
	cl::Buffer image_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					    static_cast<void *>(image.data()));
	cl::Buffer coeffs_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					     static_cast<void *>(coeffs.data()));
	cl::Buffer dDN_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					     static_cast<void *>(dDN.data()));
	cl::Buffer dDS_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					     static_cast<void *>(dDS.data()));
	cl::Buffer dDE_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					     static_cast<void *>(dDE.data()));
	cl::Buffer dDW_dev = OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buf_size,
					     static_cast<void *>(dDW.data()));
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

					set_kernel_args(r, image_dev, coeffs_dev, dDN_dev, dDS_dev, dDE_dev, dDW_dev, output_dev, M, N);
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
				set_kernel_args(r, image_dev, coeffs_dev,dDN_dev, dDS_dev, dDE_dev, dDW_dev, output_dev, M, N);
				OpenCL::executeRun<float>(*r, output_dev, gold.size(), validate);
			}
		}
	}
};

#endif // EXECUTOR_SRAD2_HARNESS_H
