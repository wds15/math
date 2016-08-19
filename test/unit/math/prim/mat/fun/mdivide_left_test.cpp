#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,mdivide_left_val) {
  using stan::math::mdivide_left;
  Eigen::MatrixXd Ad(2,2);
  Eigen::MatrixXd I;

  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_left(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}
