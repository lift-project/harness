#include <cstddef>
#include <cstdio>
#include <string>

#include "gtest/gtest.h"

#include "opencl_utils.h"
#include "mmrun.h"
#include "mm_harness.h"

struct MMTest : public ::testing::Test {

  std::string timing_filename;

  MMTest() {
    OpenCL::init(0, 0);

    timing_filename = File::get_timing_filename();
  }

  virtual void SetUp() {
    if (File::is_file_exist(timing_filename))
      std::remove(timing_filename.c_str());
  }

  virtual void TearDown() {
    if (File::is_file_exist(timing_filename))
      std::remove(timing_filename.c_str());
  }
};

TEST_F(MMTest, CanCreate) {

  auto settings = std::vector<std::string>(
      {"1024", "1024", "1024", "1", "16", "16", "1", "bla", "0", "0"});

  auto mmRun = new MMRun<float>(settings, 1024, 1024, 1024, 0, 0, 0);
  EXPECT_NE(nullptr, mmRun);
}

TEST_F(MMTest, RunSquare) {

  auto base_filename = std::string("../../resources/mm");
  auto filename = base_filename + ".cl";

  ASSERT_TRUE(File::is_file_exist(filename));

  size_t size = 256;
  auto size_string = std::to_string(size);

  auto settings =
      std::vector<std::string>({size_string, size_string, size_string, "1",
                                "16", "16", "1", base_filename, "0", "0"});

  auto mmRun = new MMRun<float>(settings, size, size, size, 0, 0, 0);
  auto run =
      std::vector<std::shared_ptr<::Run>>({std::shared_ptr<::Run>(mmRun)});
  run_harness<float>(run, size, size, size, "testFile1", "testFile2",
                     "testFile3", true, false, false, false, false, false);

  EXPECT_TRUE(File::is_file_exist(timing_filename));
}

TEST_F(MMTest, RunRectangular) {

  auto base_filename = std::string("../../resources/mm");
  auto filename = base_filename + ".cl";

  ASSERT_TRUE(File::is_file_exist(filename));

  size_t size_M = 512;
  size_t size_K = 256;
  size_t size_N = 128;

  auto settings = std::vector<std::string>(
      {"", std::to_string(size_M), std::to_string(size_N), "1", "16", "16", "1",
       base_filename, "0", "0"});

  auto mmRun = new MMRun<float>(settings, size_M, size_K, size_N, 0, 0, 0);
  auto run =
      std::vector<std::shared_ptr<::Run>>({std::shared_ptr<::Run>(mmRun)});
  run_harness<float>(run, size_M, size_K, size_N, "testFile1", "testFile2",
                     "testFile3", true, false, false, false, false, false);

  EXPECT_TRUE(File::is_file_exist(timing_filename));
}

TEST_F(MMTest, RunRectangularTransposeB) {

  auto base_filename = std::string("../../resources/mm_tb");
  auto filename = base_filename + ".cl";

  ASSERT_TRUE(File::is_file_exist(filename));

  size_t size_M = 512;
  size_t size_K = 256;
  size_t size_N = 128;

  auto settings = std::vector<std::string>(
      {"", std::to_string(size_M), std::to_string(size_N), "1", "16", "16", "1",
       base_filename, "0", "0"});

  auto mmRun = new MMRun<float>(settings, size_M, size_K, size_N, 0, 0, 0);
  auto run =
      std::vector<std::shared_ptr<::Run>>({std::shared_ptr<::Run>(mmRun)});
  run_harness<float>(run, size_M, size_K, size_N, "testFile1", "testFile2",
                     "testFile3", true, false, true, false, false, false);

  EXPECT_TRUE(File::is_file_exist(timing_filename));
}
