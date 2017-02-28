#include <string>
#include <vector>
#include <iostream>
#include <condition_variable>
#include <queue>
#include <thread>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/program_options.hpp>
#include <opencl_utils.h>

namespace po = boost::program_options;
namespace pt = boost::property_tree;

using namespace std;

vector<int> size_arguments;
vector<pair<string,long>> inputs;
size_t output_size;
vector<vector<float>> read_inputs;

void load_inputs(vector<pair<string,long>>& inputs) {

  cout << "Loading inputs..." << endl;

  read_inputs.reserve(inputs.size());

  for (auto& pair : inputs) {

    auto& filename = pair.first;
    auto& size = pair.second;

    vector<float> contents;
    auto num_elements = size / sizeof(float);
    contents.reserve(num_elements);

    ifstream in(filename);

    if (!in.good())
      return;

    while (contents.size() < num_elements) {
      float temp;
      in >> temp;

      contents.push_back(temp);
    }

    read_inputs.push_back(contents);
  }

}

void load_configuration(const string& filename) {

  cout << "Loading configuration..." << endl;

  pt::ptree tree;

  pt::read_json(filename, tree);

  for (auto& size : tree.get_child("sizes"))
    size_arguments.push_back(size.second.get_value<int>());

  output_size = tree.get<size_t>("output");

  for (auto& input : tree.get_child("inputs")) {

    string filename = input.second.get<string>("filename");
    long size = input.second.get<long>("size");

    inputs.push_back({ filename, size });

  }

  load_inputs(inputs);

}

void execute(function<bool(const vector<float>&)> validate,
    const vector<cl::Buffer> &input_buffers, const cl::Buffer &output_dev,
    shared_ptr<Run> &r);

/**
 * FIXME: This is a lazy copy paste of the old main with a template switch for single and double
 * precision
 */
void run_harness(std::vector<std::shared_ptr<Run>> &all_run,
    const bool threaded, const bool binary) {

	if (binary) cout << "Using precompiled binaries" << endl;

	// validation function
	auto validate = [&](const std::vector<float> &output) { return true; };

  // TODO: inputs?? mix of buffers and the rest

  vector<cl::Buffer> input_buffers;

  for (auto& input : read_inputs) {
    input_buffers.push_back(
        OpenCL::alloc(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            input.size() * sizeof(float), static_cast<void*>(input.data()))
    );

  }

	// Allocating buffers
	cl::Buffer output_dev = OpenCL::alloc(CL_MEM_READ_WRITE, output_size);

	// multi-threaded exec
	if (threaded) {
		mutex m;
		condition_variable cv;

		bool done = false;
		bool ready = false;
		queue<std::shared_ptr<Run>> ready_queue;

		// compilation thread
		auto compilation_thread = std::thread([&] {
			for (auto &r : all_run) {
				if (r->compile(binary)) {
					unique_lock<std::mutex> locker(m);
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
					while (!ready && !done)
						cv.wait(locker);
        }

        while (!ready_queue.empty()) {
          {
            std::unique_lock<std::mutex> locker(m);
            r = ready_queue.front();
            ready_queue.pop();
          }

          execute(validate, input_buffers, output_dev, r);
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

      if (r->compile(binary))
        execute(validate, input_buffers, output_dev, r);

    }
  }
}

void execute(function<bool(const std::vector<float>&)> validate,
    const vector<cl::Buffer> &input_buffers, const cl::Buffer &output_dev,
    shared_ptr<Run> &r) {
  cl_uint i;

  for (i = 0; i < inputs.size(); i++) {

    auto type = inputs[i].first;

    if (type.find("_") == string::npos)
      r->getKernel().setArg(i, read_inputs[i].front());
    else
      r->getKernel().setArg(i, input_buffers[i]);
  }

  r->getKernel().setArg(i, output_dev);

  OpenCL::executeRun<float>(*r, output_dev, output_size / sizeof(float), validate);
};

template<typename T>
struct GenericRun : public Run {
  size_t num_args;
  vector<int>& size_arguments;

  /**
   * Deserialize a line from the CSV
   */
  GenericRun(
      const std::vector<std::string>& values,
      vector<int>& size_arguments, size_t num_args,
      size_t default_local_0 = 1, size_t default_local_1 = 1, size_t default_local_2 = 1):
      Run(values, default_local_0, default_local_1, default_local_2),
      num_args(num_args), size_arguments(size_arguments) {}

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for (auto &size: extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t) size});

    auto idx = (cl_uint) num_args + 1;

    // Skip the first num_args to account for inputs/outputs
    for (const auto& arg: extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local: extra_local_args)
      kernel.setArg(idx++, local);

    for (const auto& size : size_arguments)
      kernel.setArg(idx++, size);
  }
};

int main(int argc, const char *const *argv) {

  string input_file;
  unsigned platform, device;

  unsigned local_0, local_1, local_2;

  bool threaded, binary;

  po::options_description description("Allowed options");

  description.add_options()
      ("help,h", "Produce this message")
      ("input,i", po::value<string>(&input_file)->required(),
          "The input configuration file.")
      ("platform,p", po::value<unsigned>(&platform)->default_value(0),
          "OpenCL platform index")
      ("device,d", po::value<unsigned>(&device)->default_value(0),
          "OpenCL device index")
      ("timeout,t", po::value<float>(&OpenCL::timeout)->default_value(100.0f),
          "Timeout to avoid multiple executions")
      ("iterations,i", po::value<int>(&OpenCL::iterations)->default_value(10),
          "Number of iterations for each experiment")
      ("local-combinations,l", po::value<bool>(&OpenCL::local_combinations)->default_value(false),
          "Run different valid combinations of local sizes instead of letting the implementation choose if the local size is marked '?'.")
      ("l0", po::value<unsigned>(&local_0)->default_value(0),
          "Local size in dim 0 to use if specified as '?'")
      ("l1", po::value<unsigned>(&local_1)->default_value(0),
          "Local size in dim 1 to use if specified as '?'")
      ("l2", po::value<unsigned>(&local_2)->default_value(0),
          "Local size in dim 2 to use if specified as '?'")
      ("min-local", po::value<size_t>(&OpenCL::min_local_size)->default_value(1),
          "The minimum local size to use when running the experiments")
      ("b,binary", po::value<bool>(&binary)->default_value(false),
          "Load programs as binaries instead of compiling OpenCL-C source.")
      ("threaded", po::value<bool>(&threaded),
          "Use a separate thread for compilation and execution")
      ;

  try {

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << description << endl;
      return 1;
    }
  } catch (std::exception& e) {
    cout << e.what() << endl;
    cout << description << endl;
    return 1;
  }

  load_configuration(input_file);

  auto size_string = to_string(size_arguments.front());

  if (size_arguments.size() > 1 &&
      !all_of(size_arguments.begin(), size_arguments.end(),
          [&](int size) { return size == size_arguments.front(); })) {

    for (auto i = 1u; i < size_arguments.size(); i++)
      size_string += "_" + to_string(size_arguments[i]);

  }

  File::set_size(size_string);


  // === Loading CSV file ===
  auto all_run = Csv::init(
      [&](const std::vector<std::string>& values) {
               return std::shared_ptr<Run>(new GenericRun<float>(values, size_arguments,
                   inputs.size(), local_0, local_1, local_2));
      });


  OpenCL::init(platform, device);

  run_harness(all_run, threaded, binary);

}
