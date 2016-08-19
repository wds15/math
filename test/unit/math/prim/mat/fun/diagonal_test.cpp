#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, diagonal) {
  Eigen::MatrixXd m0;

  using stan::math::diagonal;
  EXPECT_NO_THROW(diagonal(m0));
}
