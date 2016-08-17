//
// Created by s1042579 on 17/08/16.
//

#include <cstdio>
#include <string>

#include "gtest/gtest.h"

#include "opencl_utils.h"
#include "kmeansrun.h"
#include "kmeans_harness.h"

struct KMeansTest: public ::testing::Test {

  std::string timing_filename;

  KMeansTest() {
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

TEST_F(KMeansTest, CanCreate) {

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

  auto nBodyRun = new KMeansRun<float>(settings, 1024,5,34,0,0,0);
  EXPECT_NE(nullptr, nBodyRun);
}
TEST_F(KMeansTest, Run) {

  auto base_filename = std::string("../../resources/kmeans");
  auto filename = base_filename + ".cl";

  ASSERT_TRUE(File::is_file_exist(filename));

  size_t num_points = 256;
  size_t num_clusters = 5;
  size_t num_features = 34;
  auto size_string = std::to_string(num_points);

  auto settings = std::vector<std::string>({
      "",
      size_string,
      "1",
      "1",
      "128",
      "1",
      "1",
      base_filename,
      "0",
      "0"
  });

  auto nBodyRun = new KMeansRun<float>(settings, num_points, num_clusters, num_features, 0, 0, 0);
  auto run = std::vector<std::shared_ptr<::Run>>({std::shared_ptr<::Run>(nBodyRun)});
  run_harness<float>(
      run,
      num_points, num_clusters, num_features,
      "testFile1", "testFile2", "testFile3",
      true, false, false, true
  );

  EXPECT_TRUE(File::is_file_exist(timing_filename));
}
