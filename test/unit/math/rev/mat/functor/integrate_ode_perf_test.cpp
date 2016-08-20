#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <ctime>

#include <boost/random/mersenne_twister.hpp>

#include <stan/math.hpp>

// translation of Hornberg model to Stan, model from
// http://jjj.biochem.sun.ac.za/database/hornberg/index.html and
// described in paper DOI 10.1111/j.1432-1033.2004.04404.x
#include <test/unit/math/rev/mat/functor/hornberg.hpp>


TEST(StanOde_stiff_stress_test, cvodes_bdf_hornberg) {

  const size_t nIntegrations = 100;
  const double rel_tol = 1E-5;
  const double abs_tol = 1E-5;
  const double sdlog = 3.0;
  //const double sdlog = 0.1;
    
  hornberg_ode_fun f_;

  // initial value and parameters from model definition
  std::vector<double> y0(8);
  y0[0] = 0.5;
  y0[1] = 0.0;
  y0[2] = 1.0;
  y0[3] = 0.0;
  y0[4] = 1.0;
  y0[5] = 0.0;
  y0[6] = 1.0;
  y0[7] = 0.0;

  double t0 = 0;

  std::vector<double> ts;
  const double tmax = 100;

  for (int i = 0; i < 200; i++)
    ts.push_back((i+1)*tmax/200.);

  std::vector<double> theta(18);

  theta[0] = 1.0;
  theta[1] = 0.1;
  theta[2] = 0.01;
  theta[3] = 0.1;
  theta[4] = 1.0;
  theta[5] = 0.1;
  theta[6] = 0.3;
  theta[7] = 1.0;
  theta[8] = 1.0;
  theta[9] = 0.1;
  theta[10] = 0.3;
  theta[11] = 1.0;
  theta[12] = 1.0;
  theta[13] = 0.1;
  theta[14] = 0.3;
  theta[15] = 1.0;
  theta[16] = 0.0;
  theta[17] = 1.0;

  boost::random::mt19937 rng;

  std::vector<double> data;

  std::vector<int> data_int;

  size_t integration = 0;
  
  rng.seed(45656);
  
  for( ; integration < nIntegrations; integration++) {

    std::vector<double> theta_run(theta);

    for(size_t i = 0; i < theta.size(); i++)
      theta_run[i] *= stan::math::lognormal_rng(0, sdlog, rng);

    std::vector<double> y0_run(y0);

    for(size_t i = 0; i < y0.size(); i++)
      y0_run[i] *= stan::math::lognormal_rng(0, sdlog, rng);

    std::vector<stan::math::var> y0_var(y0_run.begin(), y0_run.end());
    std::vector<stan::math::var> theta_var(theta_run.begin(), theta_run.end());

    std::vector<std::vector<stan::math::var> > res_cvode
      = stan::math::integrate_ode_bdf(f_, y0_var, t0, ts, theta_var, data, data_int, 0, rel_tol, abs_tol);
    
    stan::math::recover_memory();
  }

  EXPECT_EQ(integration, nIntegrations);
 }


TEST(StanOde_stiff_stress_test, odeSD_hornberg) {

  const size_t nIntegrations = 100;
  const double rel_tol = 1E-5;
  const double abs_tol = 1E-5;
  const double sdlog = 3.0;
  //const double sdlog = 0.1;
  
  hornberg_ode_fun f_;
  hornberg_ode_sd_fun fp_;

  // initial value and parameters from model definition
  std::vector<double> y0(8);
  y0[0] = 0.5;
  y0[1] = 0.0;
  y0[2] = 1.0;
  y0[3] = 0.0;
  y0[4] = 1.0;
  y0[5] = 0.0;
  y0[6] = 1.0;
  y0[7] = 0.0;

  double t0 = 0;

  std::vector<double> ts;
  const double tmax = 100;

  for (int i = 0; i < 200; i++)
    ts.push_back((i+1)*tmax/200.);

  std::vector<double> theta(18);

  theta[0] = 1.0;
  theta[1] = 0.1;
  theta[2] = 0.01;
  theta[3] = 0.1;
  theta[4] = 1.0;
  theta[5] = 0.1;
  theta[6] = 0.3;
  theta[7] = 1.0;
  theta[8] = 1.0;
  theta[9] = 0.1;
  theta[10] = 0.3;
  theta[11] = 1.0;
  theta[12] = 1.0;
  theta[13] = 0.1;
  theta[14] = 0.3;
  theta[15] = 1.0;
  theta[16] = 0.0;
  theta[17] = 1.0;

  boost::random::mt19937 rng;

  std::vector<double> data;

  std::vector<int> data_int;

  size_t integration = 0;

    rng.seed(45656);
  for( ; integration < nIntegrations; integration++) {

    std::vector<double> theta_run(theta);

    for(size_t i = 0; i < theta.size(); i++)
      theta_run[i] *= stan::math::lognormal_rng(0, sdlog, rng);

    std::vector<double> y0_run(y0);

    for(size_t i = 0; i < y0.size(); i++)
      y0_run[i] *= stan::math::lognormal_rng(0, sdlog, rng);

    std::vector<stan::math::var> y0_var(y0_run.begin(), y0_run.end());
    std::vector<stan::math::var> theta_var(theta_run.begin(), theta_run.end());

    std::vector<std::vector<stan::math::var> > res_odeSD
      = stan::math::integrate_ode_odeSD(f_, fp_, y0_var, t0, ts, theta_var, data, data_int, 0, rel_tol, abs_tol);
    
    stan::math::recover_memory();
  }
  
  EXPECT_EQ(integration, nIntegrations);
 }
