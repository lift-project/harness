#include <string>
#include <vector>

#include "opencl_utils.h"
#include "run.h"

// Load the file and compile the program
bool Run::compile(bool binary_mode) {
  return OpenCL::compile(hash, kernel, loc1 * loc2 * loc3, binary_mode);
}

Run::Run(const std::vector<std::string> &values, std::size_t default_local_0,
         std::size_t default_local_1, std::size_t default_local_2) {

  assert(values.size() > 8 && "Bad CSV format");

  // Skip input size
  auto i = 1;

  // Global NDRange
  glob1 = Csv::readInt(values[i++]);
  glob2 = Csv::readInt(values[i++]);
  glob3 = Csv::readInt(values[i++]);

  // Local NDRange
  loc1 = Csv::readInt(values[i++]); // returns 0 if it could not parse the
                                    // string to an int, e.g. for '?'.
  loc2 = Csv::readInt(values[i++]);
  loc3 = Csv::readInt(values[i++]);

  if (loc1 == 0)
    loc1 = default_local_0;
  if (loc2 == 0)
    loc2 = default_local_1;
  if (loc3 == 0)
    loc3 = default_local_2;

  // Source hash
  hash = values[i++];
  hash.erase(std::remove_if(std::begin(hash), std::end(hash), isspace),
             std::end(hash));

  // Number of temporary buffers to allocate and their sizes
  auto num_buf = Csv::readInt(values[i++]);
  for (unsigned x = 0; x < num_buf; ++x)
    extra_buffer_size.push_back((int)Csv::readInt(values[i++]));

  // Number of local buffers to allocate and their sizes
  auto num_local = Csv::readInt(values[i++]);
  for (unsigned x = 0; x < num_local; ++x)
    extra_local_args.push_back({Csv::readInt(values[i++])});

  for (const auto &local_arg : extra_local_args) {
    sum_local += local_arg.size_;
  }
}
