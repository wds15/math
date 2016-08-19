#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, transpose) {
  Eigen::VectorXd v0;
  Eigen::RowVectorXd rv0;
  Eigen::MatrixXd m0;

  using stan::math::transpose;
  EXPECT_NO_THROW(transpose(v0));
  EXPECT_NO_THROW(transpose(rv0));
  EXPECT_NO_THROW(transpose(m0));
}
