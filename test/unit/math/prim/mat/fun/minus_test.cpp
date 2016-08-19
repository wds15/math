#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, minus) {
  Eigen::VectorXd v0;
  Eigen::RowVectorXd rv0;
  Eigen::MatrixXd m0;

  EXPECT_EQ(0,stan::math::minus(v0).size());
  EXPECT_EQ(0,stan::math::minus(rv0).size());
  EXPECT_EQ(0,stan::math::minus(m0).size());
}
