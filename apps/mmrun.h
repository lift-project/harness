//
// Created by s1042579 on 16/08/16.
//

#ifndef EXECUTOR_MM_H
#define EXECUTOR_MM_H

template<typename T>
struct MMRun: public Run {

  std::size_t M;
  std::size_t K;
  std::size_t N;

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
  MMRun(const std::vector<std::string>& values, std::size_t M,
      std::size_t K, std::size_t N, std::size_t default_local_0,
      std::size_t default_local_1, std::size_t default_local_2):
      M(M), K(K), N(N), default_local_0(default_local_0),
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

    cl_uint idx = 3;

    // Skip the first 3 to compensate for the csv (forgot a drop(3) in scala)
    for(const auto &arg: extra_args)
      kernel.setArg(idx++, arg);

    for (const auto &local: extra_local_args)
      kernel.setArg(idx++, local);

    // TODO: order?
    kernel.setArg(idx++, (int) K);
    kernel.setArg(idx++, (int) M);
    kernel.setArg(idx, (int) N);
  }

  void cleanup() override {
    extra_buffer_size.clear();
    kernel = cl::Kernel();
  }
};

#endif //EXECUTOR_MM_H
