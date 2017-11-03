#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <condition_variable>
#include <queue>
#include <thread>
#include <type_traits>
#include <cstdio>

#include <boost/optional.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/program_options.hpp>

#include "file_utils.h"
#include "opencl_utils.h"

namespace po = boost::program_options;
namespace pt = boost::property_tree;

using namespace std;

string input_file_folder;

bool binary_input;

vector<int> size_arguments;
vector<pair<string, size_t>> inputs;
vector<vector<float>> read_inputs;
vector<vector<char>> binary_inputs;

size_t output_size;
boost::optional<string> output_file;
vector<float> gold_output;
vector<char> binary_gold_output;

template <typename T>
void read_file_with_size(vector<T> &contents, const string &filename,
                         const size_t size) {

  auto num_elements = size / sizeof(T);
  contents.reserve(num_elements);

  ifstream in(filename);

  if (!in.good())
    throw invalid_argument("Missing input file");

  while (contents.size() < num_elements) {
    T temp;
    in >> temp;
    contents.push_back(temp);
  }
}

void load_text_inputs_and_outputs() {

  cout << "Loading text inputs..." << endl;

  read_inputs.reserve(inputs.size());

  for (auto &pair : inputs) {

    auto filename = input_file_folder + "/" + pair.first;
    auto size = pair.second;

    vector<float> contents;

    read_file_with_size(contents, filename, size);

    read_inputs.push_back(contents);
  }

  if (output_file) {
    auto filename = input_file_folder + "/" + output_file.get();
    read_file_with_size(gold_output, filename, output_size);
  }

  cout << "Inputs loaded." << endl;
}

void load_binary_inputs_and_outputs() {

  cout << "Loading binary inputs..." << endl;

  binary_inputs.reserve(inputs.size());

  for (auto &pair : inputs) {

    auto filename = input_file_folder + "/" + pair.first;
    auto size = pair.second;

    vector<char> contents(size);

    File::load_input(contents, filename);

    binary_inputs.push_back(move(contents));
  }

  if (output_file) {
    auto filename = input_file_folder + "/" + output_file.get();
    binary_gold_output = vector<char>(output_size);
    File::load_input(binary_gold_output, filename);
  }
}

void load_configuration(const string &filename) {

  cout << "Loading configuration..." << endl;

  pt::ptree tree;

  pt::read_json(filename, tree);

  for (auto &size : tree.get_child("sizes"))
    size_arguments.push_back(size.second.get_value<int>());

  output_size = tree.get<size_t>("output");
  output_file = tree.get_optional<string>("output_file");

  auto optional_binary = tree.get_optional<bool>("binary");
  binary_input = optional_binary ? optional_binary.get() : false;

  for (auto &input : tree.get_child("inputs")) {

    string filename = input.second.get<string>("filename");
    long size = input.second.get<size_t>("size");

    inputs.push_back({filename, size});
  }
}
template<typename T>
vector<T> get_output_from_binary() {
  vector<T> correct_type(binary_gold_output.size() / sizeof(T));

  memcpy(correct_type.data(), binary_gold_output.data(), binary_gold_output.size());

  return correct_type;
}

template<typename T>
vector<T> get_output() {

  if (!binary_input && output_file)
    throw invalid_argument("Only float non binary outputs are allowed");

  return get_output_from_binary<T>();
}

template<>
vector<float> get_output() {
  if (binary_input)
    return get_output_from_binary<float>();
  else
    return gold_output;
}

template<typename T>
bool compare_values(typename enable_if<is_floating_point<T>::value, T>::type x, T y) {
  if (abs(x - y) > 0.001 * max(abs(x), abs(y)))
    return false;
  else
    return true;
}

template<typename T>
bool compare_values(typename enable_if<is_integral<T>::value, T>::type x, T y) {
  return x == y;
}

template<typename T>
bool validate(const vector<T> &kernel_output, const vector<T>& expected_output) {

  if (expected_output.size() != kernel_output.size())
    return false;

  for (auto i = 0u; i < expected_output.size(); i++) {
    auto x = gold_output[i];
    auto y = kernel_output[i];

    if (!compare_values(x,y))
      return false;
  }

  return true;
}

vector<cl::Buffer> allocate_input_buffers() {

  vector<cl::Buffer> input_buffers;

  // Allocate input buffers
  if (binary_input) {
    for (auto &input : binary_inputs) {

      input_buffers.push_back(OpenCL::alloc(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size(),
            static_cast<void *>(input.data())));
    }
  } else {
  // TODO: inputs?? mix of buffers and the rest
  // TODO: Other data types
    for (auto &input : read_inputs) {
      input_buffers.push_back(OpenCL::alloc(
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float),
            static_cast<void *>(input.data())));
    }
  }

  return input_buffers;
}

void load_inputs() {
  if (binary_input)
    load_binary_inputs_and_outputs();
  else
    load_text_inputs_and_outputs();
}

template <typename T>
void execute(boost::optional<function<bool(const std::vector<T> &)>> validate,
             const vector<cl::Buffer> &input_buffers,
             const cl::Buffer &output_dev, shared_ptr<Run> &r) {
  cl_uint i;

  for (i = 0; i < inputs.size(); i++) {

    auto type = inputs[i].first;

    if (type.find("_") == string::npos)
      if (binary_input)
        r->getKernel().setArg(i, inputs[i].second, (void*) binary_inputs[i].data());
      else
        r->getKernel().setArg(i, read_inputs[i].front());
    else
      r->getKernel().setArg(i, input_buffers[i]);
  }

  r->getKernel().setArg(i, output_dev);

  OpenCL::executeRun<T>(*r, output_dev, output_size / sizeof(T), validate);
};

bool check_for_timeout(shared_ptr<Run>& r) {
  auto hash = r->hash;
  auto timeout_candidate_file = hash + ".timeout";
  auto has_timed_out = File::is_file_exist(timeout_candidate_file);

  if (has_timed_out) {
    File::add_timeout(hash);
    std::remove(timeout_candidate_file.c_str());
  } else {
    std::ofstream outfile(timeout_candidate_file);
  }

  return has_timed_out;
}

void clean_timeout(shared_ptr<Run>& r) {
  auto hash = r->hash;
  auto timeout_candidate_file = hash + ".timeout";
  std::remove(timeout_candidate_file.c_str());
}

template<typename T>
void run_harness(std::vector<std::shared_ptr<Run>> &all_run,
                 const bool threaded, const bool binary) {

  if (binary)
    cout << "Using precompiled binaries" << endl;

  load_inputs();

  auto input_buffers = allocate_input_buffers();
  auto expected_output = get_output<T>();

  boost::optional<function<bool(const vector<T> &)>> optional_validation;

  auto validation_function = [&expected_output] (const vector<T>& kernel_output) {
    return validate(kernel_output, expected_output);
  };

  if (output_file)
    optional_validation =
      boost::optional<function<bool(const vector<T> &)>>(validation_function);

  // Allocating the output buffer
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
      auto first = true;
      for (auto &r : all_run) {

        if (first && check_for_timeout(r))
          continue;

        if (r->compile(binary)) {
          unique_lock<std::mutex> locker(m);
          ready_queue.push(r);
          ready = true;
          cv.notify_one();
        }

        if (first)
          clean_timeout(r);

        first = false;
      }
    });

    auto execute_thread = std::thread([&] {
      std::shared_ptr<Run> r = nullptr;
      auto first = true;
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

          auto hash = r->hash;
          auto timeout_candidate_file = hash + ".timeout";

          if (first && check_for_timeout(r))
            continue;

          execute(optional_validation, input_buffers, output_dev, r);

          if (first)
            std::remove(timeout_candidate_file.c_str());

          first = false;
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
        execute(optional_validation, input_buffers, output_dev, r);
    }
  }
}

struct GenericRun : public Run {
  size_t num_args;
  vector<int> &size_arguments;

  /**
   * Deserialize a line from the CSV
   */
  GenericRun(const std::vector<std::string> &values,
             vector<int> &size_arguments, size_t num_args,
             size_t default_local_0 = 1, size_t default_local_1 = 1,
             size_t default_local_2 = 1)
      : Run(values, default_local_0, default_local_1, default_local_2),
        num_args(num_args), size_arguments(size_arguments) {}

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for (auto &size : extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

    auto idx = (cl_uint)num_args + 1;

    // Skip the first num_args to account for inputs/outputs
    for (const auto &arg : extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local : extra_local_args)
      kernel.setArg(idx++, local);

    for (const auto &size : size_arguments)
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
    ("file,f", po::value<string>(&input_file)->required(),
      "The input configuration file.")
    ("folder", po::value<string>(&input_file_folder)->default_value("."),
      "The folder where to look for input data files.")
    ("platform,p", po::value<unsigned>(&platform)->default_value(0),
      "OpenCL platform index")
    ("device,d", po::value<unsigned>(&device)->default_value(0),
      "OpenCL device index")
    ("timeout,t", po::value<float>(&OpenCL::timeout)->default_value(100.0f),
      "Timeout to avoid multiple executions")
    ("iterations,i", po::value<int>(&OpenCL::iterations)->default_value(10),
      "Number of iterations for each experiment")
    ("local-combinations,l", po::bool_switch(&OpenCL::local_combinations)->default_value(false),
      "Run different valid combinations of local sizes instead of letting the "
      "implementation choose if the local size is marked '?'.")
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
    ("threaded", po::value<bool>(&threaded)->default_value(true),
      "Use a separate thread for compilation and execution")
    ;

  try {

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << description << endl;
      return 1;
    }
  } catch (std::exception &e) {
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
  auto all_run = Csv::init([&](const std::vector<std::string> &values) {
    return std::shared_ptr<Run>(new GenericRun(
        values, size_arguments, inputs.size(), local_0, local_1, local_2));
  });

  if (all_run.empty()) {
    File::create_done_marker(size_string);
    exit(0);
  }

  OpenCL::init(platform, device, OpenCL::iterations);

  if (!output_file)
    run_harness<char>(all_run, threaded, binary);
  else if (output_file.get().find("float") != string::npos)
    run_harness<float>(all_run, threaded, binary);
  else if (output_file.get().find("int") != string::npos)
    run_harness<int>(all_run, threaded, binary);
  else
    throw invalid_argument("Unknown output type!");
}

