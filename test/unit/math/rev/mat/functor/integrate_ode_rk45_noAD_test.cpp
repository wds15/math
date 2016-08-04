#include <stan/math/rev/mat.hpp>
#include <stan/math/rev/core.hpp>

#include <gtest/gtest.h>

#include <vector>

#include <boost/numeric/odeint.hpp>

#include <test/unit/math/rev/mat/functor/oral_2cmt.hpp>
// load specliased template def to avoid usage of AD
#include <test/unit/math/rev/mat/functor/oral_2cmt_jacobian.hpp>

struct plain_oral_2cmt_ode_fun : public oral_2cmt_ode_fun {};

#include <test/unit/util.hpp>

// test which will not use AD to construct the coupled system
TEST(StanOde_noAD_test, odeint_2cmt_oral_fast) {

  oral_2cmt_ode_fun f_;

  // initial value and parameters from model definition
  std::vector<stan::math::var> y0_v(3);
  y0_v[0] = 1E8;
  y0_v[1] = 0;
  y0_v[2] = 0;

  double t0 = 0;

  std::vector<double> ts_long;
  ts_long.push_back(1E4);
  ts_long.push_back(1E5);

  std::vector<stan::math::var> theta_v(4);

  theta_v[0] = log(2.)/1;
  theta_v[1] = log(2.)/70.;
  theta_v[2] = log(2.)/20.;
  theta_v[3] = log(2.)/50.;

  std::vector<double> data;

  std::vector<int> data_int;

  stan::math::integrate_ode_rk45(f_, y0_v, t0, ts_long, theta_v, data, data_int, 0, 1E-10, 1E-10, 1000000);
}

// test which will use AD to construct the coupled system
TEST(StanOde_noAD_test, odeint_2cmt_oral_slow) {

  plain_oral_2cmt_ode_fun f_;

  // initial value and parameters from model definition
  std::vector<stan::math::var> y0_v(3);
  y0_v[0] = 1E8;
  y0_v[1] = 0;
  y0_v[2] = 0;

  double t0 = 0;

  std::vector<double> ts_long;
  ts_long.push_back(1E4);
  ts_long.push_back(1E5);

  std::vector<stan::math::var> theta_v(4);

  theta_v[0] = log(2.)/1;
  theta_v[1] = log(2.)/70.;
  theta_v[2] = log(2.)/20.;
  theta_v[3] = log(2.)/50.;

  std::vector<double> data;

  std::vector<int> data_int;

  stan::math::integrate_ode_rk45(f_, y0_v, t0, ts_long, theta_v, data, data_int, 0, 1E-10, 1E-10, 1000000);
}
