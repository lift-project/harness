//
// Created by s1042579 on 17/08/16.
//

#include <cstdio>
#include <string>

#include "gtest/gtest.h"

#include "opencl_utils.h"
#include "mvrun.h"
#include "mv_harness.h"

struct MVTest: public ::testing::Test {

  std::string timing_filename;

  MVTest() {
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

TEST_F(MVTest, CanCreate) {

  auto settings = std::vector<std::string>({
      "",
      "1024",
      "",
      "1",
      "16",
      "",
      "1",
      "bla",
      "0",
      "0"
  });

  auto mvRun = new MVRun<float>(settings, 1024,0,0,0);
  EXPECT_NE(nullptr, mvRun);
}

TEST_F(MVTest, RunSquare) {

  auto base_filename = std::string("../../resources/mv");
  auto filename = base_filename + ".cl";

  ASSERT_TRUE(File::is_file_exist(filename));

  size_t size = 256;
  auto size_string = std::to_string(size);

  auto settings = std::vector<std::string>({
      "",
      size_string,
      "1",
      "1",
      "16",
      "1",
      "1",
      base_filename,
      "0",
      "0"
  });

  auto mvRun = new MVRun<float>(settings, size, 0, 0, 0);
  auto run = std::vector<std::shared_ptr<::Run>>({std::shared_ptr<::Run>(mvRun)});
  run_harness(
      run,
      size,
      "testFile1", "testFile2", "testFile3",
      true, false, false, false
  );

  EXPECT_TRUE(File::is_file_exist(timing_filename));
}
