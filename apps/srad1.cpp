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
#include "convolution2Drun.h"
#include "srad1_harness.h"
#include "options.h"

int main(int argc, char *argv[]) {
	OptParser op("Harness for Rodinias' SRAD Benchmark - Kernel1");

	auto opt_platform =
	    op.addOption<unsigned>({'p', "platform", "OpenCL platform index (default 0).", 0});
	auto opt_device =
	    op.addOption<unsigned>({'d', "device", "OpenCL device index (default 0).", 0});
	auto opt_iterations = op.addOption<unsigned>(
	    {'i', "iterations", "Execute each kernel 'iterations' times (default 10).", 10});

	auto opt_size_m =
	    op.addOption<std::size_t>({'m', "size-m", "M - number of rows (default 8194).", 8194});
	auto opt_size_n = op.addOption<std::size_t>(
	    {'n', "size-n", "N - number of columns (default 8194).", 8194});

	auto opt_binary = op.addOption<bool>(
	    {'b', "binary", "Load programs as binaries instead of compiling OpenCL-C source.",
	     false});
	auto opt_timeout = op.addOption<float>(
	    {'t', "timeout", "Timeout to avoid multiple executions (default 100ms).", 100.0f});
	auto opt_double = op.addOption<bool>({0, "double", "Use double precision.", false});
	auto opt_threaded = op.addOption<bool>(
	    {'t', "threaded", "Use a separate thread for compilation and execution (default true).",
	     true});
	auto opt_force =
	    op.addOption<bool>({'f', "force", "Override cached cross validation files.", false});
	auto opt_clean =
	    op.addOption<bool>({'c', "clean", "Clean temporary files and exit.", false});
	op.parse(argc, argv);

	using namespace std;

	// Option handling
	const size_t M = opt_size_m->get();
	const size_t N = opt_size_n->get();

	// size string used for all .csv files,
	// e.g., exec_[size_string].csv
	auto size_string = to_string(M) + "_" + to_string(N);
	if (M == N) {
		size_string = to_string(M);
	}

	File::set_size(size_string);

	OpenCL::timeout = opt_timeout->get();

	// temporary files
	std::string gold_file = "/tmp/lift_srad1_gold_" + size_string;
	std::string image_file = "/tmp/lift_srad1_image" + size_string;

	if (opt_clean->get()) {
		std::cout << "Cleaning..." << std::endl;
		for (const auto &file : {gold_file, image_file})
			std::remove(file.data());
		return 0;
	}

	// === Loading exec CSV file ===
	std::vector<std::shared_ptr<Run>> all_run =
	    Csv::init([&](const std::vector<std::string> &values) -> std::shared_ptr<Run> {
		    return std::shared_ptr<Run>(new Convolution2DRun(values, M, N));
	    });
	if (all_run.size() == 0) return 0;

	// === OpenCL init ===
	OpenCL::init(opt_platform->get(), opt_device->get(), opt_iterations->get());

	// run the harness
	run_harness(all_run, M, N, image_file, gold_file, opt_force->get(),
		    opt_threaded->get(), opt_binary->get());
}
