// [standard includes]
#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>

// [external includes]
#include <opencl_utils.h>

// [local includes]
#include "options.h"
#include "run.h"

template<typename T>
struct NBodyRun: public Run {

  std::size_t size;

  std::size_t default_local_0;
  std::size_t default_local_1;
  std::size_t default_local_2;

  // list of additional buffers to allocate
  std::vector<int> extra_buffer_size;
  std::vector<cl::Buffer> extra_args;

  // list of additional local buffers to allocate
  std::vector<cl::LocalSpaceArg> extra_local_args;

  /**
   * Deserialize a line from the CSV
   */
  NBodyRun(const std::vector<std::string>& values, std::size_t size,
      std::size_t default_local_0,
      std::size_t default_local_1, std::size_t default_local_2):
      size(size), default_local_0(default_local_0),
      default_local_1(default_local_1), default_local_2(default_local_2) {

    assert(values.size() > 8 && "Bad CSV format");

    // Skip input size
    auto i = 1;

    // global NDRange
    glob1 = Csv::readInt(values[i++]);
    glob2 = Csv::readInt(values[i++]);
    glob3 = Csv::readInt(values[i++]);

    // local NDRange
    loc1 = Csv::readInt(values[i++]); // returns 0 if it could not parse the string to an int, e.g. for '?'.
    loc2 = Csv::readInt(values[i++]);
    loc3 = Csv::readInt(values[i++]);

    if (loc1 == 0)
      loc1 = default_local_0;
    if (loc2 == 0)
      loc2 = default_local_1;
    if (loc3 == 0)
      loc3 = default_local_2;

    // source hash
    hash = values[i++];
    hash.erase(std::remove_if(std::begin(hash), std::end(hash), isspace), std::end(hash));

    // number of temporary buffers to allocate and their sizes
    auto num_buf = Csv::readInt(values[i++]);
    for(unsigned x = 0; x < num_buf; ++x)
      extra_buffer_size.push_back((int)Csv::readInt(values[i++]));

    // number of local buffers to allocate and their sizes
    auto num_local = Csv::readInt(values[i++]);
    for (unsigned x = 0; x < num_local; ++x)
      extra_local_args.push_back({Csv::readInt(values[i++])});
  }

  void setup(cl::Context context) override {
    // Allocate extra buffers
    for(auto &size: extra_buffer_size)
      extra_args.push_back({context, CL_MEM_READ_WRITE, (size_t)size});

    cl_uint idx = 5;

    // Skip the first 3 to compensate for the csv (forgot a drop(3) in scala)
    for(const auto &arg: extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local: extra_local_args)
      kernel.setArg(idx++, local);

    kernel.setArg(idx++, (int) size);
  }

  void cleanup() override {
    extra_buffer_size.clear();
    kernel = cl::Kernel();
  }
};



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
    std::uniform_real_distribution<T> distribution(0.0,1.0);

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

int main(int argc, char *argv[]) {
  OptParser op("Harness for nbody simulation.");

  // OpenCL options
  auto opt_platform = op.addOption<unsigned>({'p', "platform",
      "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>({'d', "device",
      "OpenCL device index (default 0).", 0});


  // Common options
  auto opt_binary = op.addOption<bool>({'b', "binary", "Load programs as binaries instead of compiling OpenCL-C source.", false});
  auto opt_timeout = op.addOption<float>({'t', "timeout", "Timeout to avoid multiple executions (default 100ms).", 100.0f});
  auto opt_double = op.addOption<bool>({0, "double", "Use double precision.", false});
  auto opt_threaded = op.addOption<bool>({0, "threaded", "Use a separate thread for compilation and execution (default true).", true});
  auto opt_force = op.addOption<bool>({'f', "force", "Override cached cross validation files.", false});
  auto opt_clean = op.addOption<bool>({'c', "clean", "Clean temporary files and exit.", false});

  auto opt_iterations = op.addOption<int>({'i', "iterations",
      "The number of iterations for each experiment (default 10)", 10});
  auto opt_local_combinations = op.addOption<bool>({'l', "local-combinations",
      "Run different valid combinations of local sizes instead of letting the implementation choose if the local size is marked '?'.", false});
  auto opt_local_0 = op.addOption<std::size_t>({0, "l0",
      "Local size in dim 0 to use if specified as '?'", 0});
  auto opt_local_1 = op.addOption<std::size_t>({0, "l1",
      "Local size in dim 1 to use if specified as '?'", 0});
  auto opt_local_2 = op.addOption<std::size_t>({0, "l2",
      "Local size in dim 2 to use if specified as '?'", 0});
  auto opt_min_local_size = op.addOption<std::size_t>({0, "min-local",
      "The minimum local size to use when running the experiments (defaults 1).", 1});


  // N-Body options
  auto opt_size = op.addOption<std::size_t>({'s', "size",
      "Matrix size (default 1024).", 1024});
  auto opt_espSqr = op.addOption<float>({ 'e', "espSqr", "espSqr", 500.0f });
  auto opt_deltaT = op.addOption<float>({ 0, "deltaT", "Timestep.", 0.005f });

  op.parse(argc, argv);

  using namespace std;

  // Option handling
  const size_t common_size = opt_size->get();
  auto espSqr = opt_espSqr->get();
  auto deltaT = opt_deltaT->get();

  size_t local_0 = opt_local_0->get();
  size_t local_1 = opt_local_1->get();
  size_t local_2 = opt_local_2->get();

  auto size_string = to_string(common_size);

  File::set_size(size_string);
  OpenCL::timeout = opt_timeout->get();
  OpenCL::local_combinations = opt_local_combinations->get();
  OpenCL::min_local_size = opt_min_local_size->get();
  OpenCL::iterations = opt_iterations->get();

  // Result files
  auto gold_file = "/tmp/apart_nbody_gold_" + size_string;
  auto positions_file = "/tmp/apart_nbody_positions_" + size_string;
  auto velocities_file = "/tmp/apart_nbody_velocities_" + size_string;

  if(*opt_clean) {
    cout << "Cleaning..." << endl;
    for(const auto& file: {gold_file, velocities_file})
      remove(file.data());
    return 0;
  }

  // === Loading CSV file ===
  auto all_run = Csv::init(
      [&](const std::vector<std::string>& values) -> std::shared_ptr<Run> {
        return (opt_double->get() ?

               std::shared_ptr<Run>(new NBodyRun<double>(values, common_size,
                   local_0, local_1, local_2)) :
               std::shared_ptr<Run>(new NBodyRun<float>(values,
                   common_size, local_0, local_1, local_2)));
      });

  if (all_run.size() == 0) return 0;

  // === OpenCL init ===
  OpenCL::init(opt_platform->get(), opt_device->get());

  // run the harness
  if (opt_double->get())
    run_harness<double>(
        all_run, common_size, espSqr, deltaT,
        positions_file, velocities_file, gold_file,
        opt_force->get(),
        opt_threaded->get(), opt_binary->get()
    );
  else
    run_harness<float>(
        all_run, common_size, espSqr, deltaT,
        positions_file, velocities_file, gold_file,
        opt_force->get(),
        opt_threaded->get(), opt_binary->get()
    );
}

