//
// Created by s1042579 on 16/08/16.
//

#include <cstddef>
#include "gtest/gtest.h"
#include "mmrun.h"

TEST(MMTest, CanCreate) {
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
  auto m = new MMRun<float>(settings, 1024,1024,1024,0,0,0);
  EXPECT_NE(nullptr, m);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
