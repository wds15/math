#ifndef STAN_MATH_REV_MAT_FUNCTOR_INTEGRATE_ODE_ODESD_HPP
#define STAN_MATH_REV_MAT_FUNCTOR_INTEGRATE_ODE_ODESD_HPP

#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/arr/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/err/check_ordered.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/rev/arr/fun/decouple_ode_states.hpp>
#include <stan/math/rev/mat/functor/odeSD_model.hpp>
#include <ostream>
#include <vector>

#include <odeSD/Options>
#include <odeSD/SensitivitySolver>
#include <odeSD/Solver>

namespace stan {

  namespace math {

    /**
     * Return the solutions for the specified system of ordinary
     * differential equations given the specified initial state,
     * initial times, times of desired solution, and parameters and
     * data, writing error and warning messages to the specified
     * stream.
     *
     * <b>Warning:</b> If the system of equations is stiff, roughly
     * defined by having varying time scales across dimensions, then
     * this solver is likely to be slow.
     *
     * This function is templated to allow the initial times to be
     * either data or autodiff variables and the parameters to be data
     * or autodiff variables.  The autodiff-based implementation for
     * reverse-mode are defined in namespace <code>stan::math</code>
     * and may be invoked via argument-dependent lookup by including
     * their headers.
     *
     * This function uses the <a
     * href="http://en.wikipedia.org/wiki/Dormand–Prince_method">Dormand-Prince
     * method</a> as implemented in Boost's <code>
     * boost::numeric::odeint::runge_kutta_dopri5</code> integrator.
     *
     * @tparam F type of ODE system function.
     * @tparam T1 type of scalars for initial values.
     * @tparam T2 type of scalars for parameters.
     * @param[in] f functor for the base ordinary differential equation.
     * @param[in] y0 initial state.
     * @param[in] t0 initial time.
     * @param[in] ts times of the desired solutions, in strictly
     * increasing order, all greater than the initial time.
     * @param[in] theta parameter vector for the ODE.
     * @param[in] x continuous data vector for the ODE.
     * @param[in] x_int integer data vector for the ODE.
     * @param[in, out] msgs the print stream for warning messages.
     * @return a vector of states, each state being a vector of the
     * same size as the state variable, corresponding to a time in ts.
     */
    template <typename F, typename FP, typename T1, typename T2>
    std::vector<std::vector<typename stan::return_type<T1, T2>::type> >
    integrate_ode_odeSD(const F& f,
			const FP& fp,
			const std::vector<T1>& y0,
			const double t0,
			const std::vector<double>& ts,
			const std::vector<T2>& theta,
			const std::vector<double>& x,
			const std::vector<int>& x_int,
                        std::ostream* msgs = 0,
			double relative_tolerance = 1e-10,
			double absolute_tolerance = 1e-10,
			long int max_num_steps = 1e8) {  // NOLINT(runtime/int)

      // TODO: ENABLE NON-NEGATIVE STATES OPTION !!!

      // BUT: maybe do not support features from odeSD which cannot be
      // fullfilled by the BDF integrator? This way we could provide
      // the integrate_ode_bdf result whenever somone chooses to not
      // use LGPL code or odeSD is ceasing to exist for whatever
      // reason?
      
      stan::math::check_finite("integrate_ode_odsSD", "initial state", y0);
      stan::math::check_finite("integrate_ode_odsSD", "initial time", t0);
      stan::math::check_finite("integrate_ode_odsSD", "times", ts);
      stan::math::check_finite("integrate_ode_odsSD", "parameter vector", theta);
      stan::math::check_finite("integrate_ode_odsSD", "continuous data", x);

      stan::math::check_nonzero_size("integrate_ode_odsSD", "times", ts);
      stan::math::check_nonzero_size("integrate_ode_odsSD", "initial state", y0);
      stan::math::check_ordered("integrate_ode_odsSD", "times", ts);
      stan::math::check_less("integrate_ode_odsSD", "initial time", t0, ts[0]);
      if (relative_tolerance <= 0)
        invalid_argument("integrate_ode_bdf",
                         "relative_tolerance,", relative_tolerance,
                         "", ", must be greater than 0");
      if (absolute_tolerance <= 0)
        invalid_argument("integrate_ode_bdf",
                         "absolute_tolerance,", absolute_tolerance,
                         "", ", must be greater than 0");
      if (max_num_steps <= 0)
        invalid_argument("integrate_ode_bdf",
                         "max_num_steps,", max_num_steps,
                         "", ", must be greater than 0");

      odeSD::Options options;
      options.absoluteTolerance = absolute_tolerance;
      options.relativeTolerance = relative_tolerance;
      options.maxSteps = max_num_steps;
      options.normControl = false;
      options.maxNewtonIterations = 5;

      // first time in the vector must be time of the initial state
      std::vector<double> ts_vec(ts.size() + 1);
      ts_vec[0] = t0;
      std::copy(ts.begin(), ts.end(), ts_vec.begin()+1);
      
      const size_t N = y0.size();

      std::vector<double>    y0_dbl(value_of(y0));
      std::vector<double> theta_dbl(value_of(theta));

      odeSD_model<F, FP, T1, T2> model(f, fp,
				       y0_dbl, theta_dbl,
				       x, x_int,
				       msgs,
				       ts_vec);

      if(model.size() == N) {
	odeSD::Solver odeSD_integrator(model, options);
	odeSD_integrator.integrate();
      } else {
	odeSD::SensitivitySolver odeSD_integrator(model, options);
	odeSD_integrator.integrate();
      }
      
      return decouple_ode_states(model.x_var, y0, theta);
    }
  }

}

#endif
