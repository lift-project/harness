#include <cstddef>
#include <cstdio>
#include <string>

#include "gtest/gtest.h"

#include "opencl_utils.h"
#include "mmrun.h"
#include "mm_harness.h"

struct MMTest : public ::testing::Test {
  MMTest() {
    OpenCL::init(0, 0);
  }
};

TEST_F(MMTest, CanCreate) {

  auto settings = std::vector<std::string>({
      "1024",
      "1024",
      "1024",
      "1",
      "16",
      "16",
      "1",
      "bla",
      "0",
      "0"
  });

  auto mmRun = new MMRun<float>(settings, 1024,1024,1024,0,0,0);
  EXPECT_NE(nullptr, mmRun);
}

TEST_F(MMTest, RunSquare) {

  auto base_filename = std::string("../../mm");
  auto filename = base_filename + ".cl";

  ASSERT_TRUE(File::is_file_exist(filename));

  auto timing_filename = File::get_timing_filename();

  if (File::is_file_exist(timing_filename))
    std::remove(timing_filename.c_str());

  size_t size = 256;
  auto size_string = std::to_string(size);

  auto settings = std::vector<std::string>({
      size_string,
      size_string,
      size_string,
      "1",
      "16",
      "16",
      "1",
      base_filename,
      "0",
      "0"
  });

  auto mmRun = new MMRun<float>(settings, size, size, size, 0, 0, 0);
  auto run = std::vector<std::shared_ptr<::Run>>({std::shared_ptr<::Run>(mmRun)});
  run_harness<float>(
      run,
      size, size, size,
      "testFile1", "testFile2", "testFile3",
      true, false, false, false, false, false
  );

  EXPECT_TRUE(File::is_file_exist(timing_filename));

  if (File::is_file_exist(timing_filename))
    std::remove(timing_filename.c_str());

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
