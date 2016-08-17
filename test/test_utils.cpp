//
// Created by s1042579 on 16/08/16.
//

#include "gtest/gtest.h"

#include "utils.h"


TEST(TestUtils, Transpose1) {

  // 3 rows, 3 cols
  Matrix<float> m({1,2,3,4,5,6,7,8,9});
  Matrix<float> gold({1,4,7,2,5,8,3,6,9});

  transpose(m, 3, 3);

  ASSERT_EQ(gold, m);
}

TEST(TestUtils, Transpose2) {

  // 3 rows, 4 cols
  Matrix<float> m({1,2,3,4,5,6,7,8,9,10,11,12});
  Matrix<float> gold({1,5,9,2,6,10,3,7,11,4,8,12});

  transpose(m, 3, 4);

  ASSERT_EQ(gold, m);
}

TEST(TestUtils, Transpose3) {

  // 4 rows, 3 cols
  Matrix<float> m({1,2,3,4,5,6,7,8,9,10,11,12});
  Matrix<float> gold({1,4,7,10,2,5,8,11,3,6,9,12});

  transpose(m, 4, 3);

  ASSERT_EQ(gold, m);
}


