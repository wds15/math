#pragma once

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>

struct hornberg_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline
  std::vector<typename stan::return_type<T1,T2>::type>
  operator()(const T0& t_in, // initial time
             const std::vector<T1>& x, //initial positions
             const std::vector<T2>& p, // parameters
             const std::vector<double>& sx, // double data
             const std::vector<int>& sx_int,
             std::ostream* msgs) const { // integer data
    //using std::pow;
    std::vector<typename stan::return_type<T1,T2>::type> f(8);
    f[0] = (p[0] * x[0] * -1.0) / (p[1] + x[0]) + (p[2] * x[1]) / (p[3] + x[1]);
    f[1] = (p[0] * x[0]) / (p[1] + x[0]) - (p[2] * x[1] * 1.0) / (p[3] + x[1]);
    f[2] = (p[6] * x[3]) / (p[7] + x[3]) - (p[4] * x[0] * x[2] * 1.0) / (p[5] + x[2]);
    f[3] = (p[6] * x[3] * -1.0) / (p[7] + x[3]) + (p[4] * x[0] * x[2]) / (p[5] + x[2]);
    f[4] = (p[10] * x[5]) / (p[11] + x[5]) - (p[8] * x[3] * x[4] * 1.0) / (p[9] + x[4]);
    f[5] = (p[10] * x[5] * -1.0) / (p[11] + x[5]) + (p[8] * x[3] * x[4]) / (p[9] + x[4]);
    f[6] = (p[14] * x[7]) / (p[15] + x[7] + (p[15] * p[16]) / p[17]) - (p[12] * x[5] * x[6] * 1.0) / (p[13] + x[6]);
    f[7] = (p[14] * x[7] * -1.0) / (p[15] + x[7] + (p[15] * p[16]) / p[17]) + (p[12] * x[5] * x[6]) / (p[13] + x[6]);
    return(f);
  }
};


struct hornberg_ode_sd_fun {
  template <typename T0, typename T1, typename T2>
  inline
  std::vector<typename stan::return_type<T1,T2>::type>
  operator()(const T0& t_in, // time
             const std::vector<T1>& x, // positions
	     const std::vector< typename stan::return_type<T1,T2>::type >& f, // x'
             const std::vector<T2>& p, // parameters
             const std::vector<double>& sx, // double data
             const std::vector<int>& sx_int,
             std::ostream* msgs) const { // integer data
    //using stan::math::pow;
    using std::pow;
    std::vector<typename stan::return_type<T1,T2>::type> dfdt(8);
    dfdt[0] = 1.0 / pow(p[1] + x[0], 2.0)*f[0] * p[0] * p[1] * -1.0 + 1.0 / pow(p[3] + x[1], 2.0)*f[1] * p[2] * p[3];
    dfdt[1] = 1.0 / pow(p[1] + x[0], 2.0)*f[0] * p[0] * p[1] - 1.0 / pow(p[3] + x[1], 2.0)*f[1] * p[2] * p[3] * 1.0;
    dfdt[2] = 1.0 / pow(p[7] + x[3], 2.0)*f[3] * p[6] * p[7] - (f[0] * p[4] * x[2] * 1.0) / (p[5] + x[2]) - 1.0 / pow(p[5] + x[2], 2.0)*f[2] * p[4] * p[5] * x[0] * 1.0;
    dfdt[3] = 1.0 / pow(p[7] + x[3], 2.0)*f[3] * p[6] * p[7] * -1.0 + (f[0] * p[4] * x[2]) / (p[5] + x[2]) + 1.0 / pow(p[5] + x[2], 2.0)*f[2] * p[4] * p[5] * x[0];
    dfdt[4] = 1.0 / pow(p[11] + x[5], 2.0)*f[5] * p[10] * p[11] - (f[3] * p[8] * x[4] * 1.0) / (p[9] + x[4]) - 1.0 / pow(p[9] + x[4], 2.0)*f[4] * p[8] * p[9] * x[3] * 1.0;
    dfdt[5] = 1.0 / pow(p[11] + x[5], 2.0)*f[5] * p[10] * p[11] * -1.0 + (f[3] * p[8] * x[4]) / (p[9] + x[4]) + 1.0 / pow(p[9] + x[4], 2.0)*f[4] * p[8] * p[9] * x[3];
    dfdt[6] = (f[5] * p[12] * x[6] * -1.0) / (p[13] + x[6]) - 1.0 / pow(p[13] + x[6], 2.0)*f[6] * p[12] * p[13] * x[5] * 1.0 + f[7] * p[14] * p[15] * p[17] * (p[16] + p[17])*1.0 / pow(p[17] * x[7] + p[15] * p[16] + p[15] * p[17], 2.0);
    dfdt[7] = (f[5] * p[12] * x[6]) / (p[13] + x[6]) + 1.0 / pow(p[13] + x[6], 2.0)*f[6] * p[12] * p[13] * x[5] - f[7] * p[14] * p[15] * p[17] * (p[16] + p[17])*1.0 / pow(p[17] * x[7] + p[15] * p[16] + p[15] * p[17], 2.0);
    return(dfdt);
  }
};

