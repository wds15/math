#ifndef TEST_UNIT_MATH_REV_MAT_FUNCTOR_ORAL_2CMT_HPP
#define TEST_UNIT_MATH_REV_MAT_FUNCTOR_ORAL_2CMT_HPP

#include <stan/math/rev/core.hpp>

struct oral_2cmt_ode_fun {
  template <typename T0, typename T1, typename T2>
  inline
  std::vector<typename stan::return_type<T1,T2>::type>
  operator()(const T0& t_in, // initial time
             const std::vector<T1>& y, //initial positions
             const std::vector<T2>& parms, // parameters
             const std::vector<double>& sx, // double data
             const std::vector<int>& sx_int,
             std::ostream* msgs) const { // integer data
    std::vector<typename stan::return_type<T1,T2>::type> dydt(3);

    const T2 ka  = parms[0];
    const T2 k10 = parms[1];
    const T2 k12 = parms[2];
    const T2 k21 = parms[3];

    dydt[0] = -ka * y[0];
    dydt[2] = k12 * y[1] - k21 * y[2];
    dydt[1] = -dydt[0] - k10 * y[1] -dydt[2];

    return(dydt);
  }
};

#endif
