#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, trace) {
  using stan::math::trace;
  Eigen::MatrixXd m;
  EXPECT_FLOAT_EQ(0.0,trace(m));
  m = Eigen::MatrixXd(1,1);
  m << 2.3;
  EXPECT_FLOAT_EQ(2.3,trace(m));
  m = Eigen::MatrixXd(2,3);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_FLOAT_EQ(6.0,trace(m));
}
