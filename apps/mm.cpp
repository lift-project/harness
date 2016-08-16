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

// [external includes]
#include <opencl_utils.h>

// [local includes]
#include "options.h"
#include "mmrun.h"
#include "utils.h"

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
    if(gold.size() != output.size()) return false;
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


int main(int argc, char *argv[]) {
  OptParser op("Harness for simple matrix-matrix multiply.");

  auto opt_platform = op.addOption<unsigned>({'p', "platform",
      "OpenCL platform index (default 0).", 0});
  auto opt_device = op.addOption<unsigned>({'d', "device",
      "OpenCL device index (default 0).", 0});

  auto opt_size = op.addOption<std::size_t>({'s', "size",
      "Matrix size (default 1024).", 1024});
  auto opt_size_k = op.addOption<std::size_t>({'k', "size-k",
      "Matrix size in the common dimension (default 1024).", 1024});
  auto opt_size_m = op.addOption<std::size_t>({'m', "size-m",
      "Size of matrix A in the non-common dimension (default 1024).", 1024});
  auto opt_size_n = op.addOption<std::size_t>({'n', "size-n",
      "Size of matrix B in the non-common dimension (default 1024).", 1024});
  auto opt_transposeA = op.addOption<bool>({0, "transpose-A",
      "Transpose the first matrix before computation.", false});
  auto opt_transposeB = op.addOption<bool>({0, "transpose-B",
      "Transpose the second matrix before computation.", false});
  auto opt_transposeRes = op.addOption<bool>({0, "transpose-res",
      "Transpose the output before cross validation.", false});

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

  op.parse(argc, argv);

  using namespace std;

  // Option handling
  const size_t common_size = opt_size->get();
  size_t N = opt_size_n->get();
  size_t M = opt_size_m->get();
  size_t K = opt_size_k->get();

  size_t local_0 = opt_local_0->get();
  size_t local_1 = opt_local_1->get();
  size_t local_2 = opt_local_2->get();

  auto size_string = to_string(K) + "_" + to_string(M) + "_" + to_string(N);

  if (M == 1024 && K == 1024 && N == 1024) {
    N = common_size;
    K = common_size;
    M = common_size;
    size_string = to_string(common_size);
  }

  File::set_size(size_string);
  OpenCL::timeout = opt_timeout->get();
  OpenCL::local_combinations = opt_local_combinations->get();
  OpenCL::min_local_size = opt_min_local_size->get();
  OpenCL::iterations = opt_iterations->get();

  // Result files
  auto gold_file = "/tmp/apart_mm_gold_" + size_string;
  auto matA_file = "/tmp/apart_mm_A_" + size_string;
  auto matB_file = "/tmp/apart_mm_B_" + size_string;

  if(*opt_clean) {
    cout << "Cleaning..." << endl;
    for(const auto& file: {gold_file, matA_file, matB_file})
      remove(file.data());
    return 0;
  }

  // === Loading CSV file ===
  auto all_run = Csv::init(
      [&](const std::vector<std::string>& values) -> std::shared_ptr<Run> {
        return (opt_double->get() ?

               std::shared_ptr<Run>(new MMRun<double>(values, M, K, N,
                   local_0, local_1, local_2)) :
               std::shared_ptr<Run>(new MMRun<float>(values, M, K, N,
                   local_0, local_1, local_2)));
      });

  if (all_run.size() == 0) return 0;

  // === OpenCL init ===
  OpenCL::init(opt_platform->get(), opt_device->get());

  // run the harness
  if (opt_double->get())
    run_harness<double>(
        all_run, M, K, N,
        matA_file, matB_file, gold_file,
        opt_force->get(),
        opt_transposeA->get(), opt_transposeB->get(), opt_transposeRes->get(),
        opt_threaded->get(), opt_binary->get()
    );
  else
    run_harness<float>(
        all_run, M, K, N,
        matA_file, matB_file, gold_file,
        opt_force->get(),
        opt_transposeA->get(), opt_transposeB->get(), opt_transposeRes->get(),
        opt_threaded->get(), opt_binary->get()
    );
}

