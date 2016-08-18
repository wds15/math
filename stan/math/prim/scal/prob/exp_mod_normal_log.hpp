#ifndef STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_LOG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_EXP_MOD_NORMAL_LOG_HPP

#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale,
              typename T_inv_scale>
    typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                       const T_inv_scale& lambda) {
      static const char* function("exp_mod_normal_log");

      using std::exp;
      using std::log;
      using std::sqrt;
      using boost::math::erfc;

      static const double neg_two_over_sqrt_pi
        = -2 / boost::math::constants::pi<double>();
      static const double log2 = log(2);

      typedef typename partials_return_type<T_y, T_loc, T_scale,
                                            T_inv_scale>::type
        partials_return_t;

      // check if any vectors are zero length
      if (!(stan::length(y) && stan::length(mu) && stan::length(sigma)
            && stan::length(lambda)))
        return 0;

      // validate args (here done over var, which should be OK)
      check_not_nan(function, "Random variable", y);
      check_finite(function, "Location parameter", mu);
      check_positive_finite(function, "Inv_scale parameter", lambda);
      check_positive_finite(function, "Scale parameter", sigma);
      check_consistent_sizes(function, "Random variable", y,
                             "Location parameter", mu,
                             "Scale parameter", sigma,
                             "Inv_scale paramter", lambda);

      // check if no variables are involved and prop-to
      if (!include_summand<propto, T_y, T_loc, T_scale, T_inv_scale>::value)
        return 0;

      // set up return value accumulator
      partials_return_t logp(0.0);

      // set up template expressions wrapping scalars into vector views
      OperandsAndPartials<T_y, T_loc, T_scale, T_inv_scale>
        operands_and_partials(y, mu, sigma, lambda);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_loc> mu_vec(mu);
      VectorView<const T_scale> sigma_vec(sigma);
      VectorView<const T_inv_scale> lambda_vec(lambda);
      size_t N = max_size(y, mu, sigma, lambda);

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments (not nec. double, despite name)
        partials_return_t y_dbl = value_of(y_vec[n]);
        partials_return_t mu_dbl = value_of(mu_vec[n]);
        partials_return_t sigma_dbl = value_of(sigma_vec[n]);
        partials_return_t lambda_dbl = value_of(lambda_vec[n]);

        // cache values used more than once
        partials_return_t sqrt2_sigma = sqrt2 * sigma_dbl;
        partials_return_t sqrt2_sigma_sq = square(sqrt2_sigma);
        partials_return_t mu_minus_y = mu_dbl - y_dbl;
        partials_return_t sigma_sq = square(sigma_dbl);
        partials_return_t lambda_sigma_sq = lambda_dbl * sigma_sq;
        partials_return_t mu_minus_y_plus_lambda_sigma_sq_over_sqrt2_sigma
          = (mu_minus_y + lambda_sigma_sq) / sqrt2_sigma;

        // log probability
        if (include_summand<propto>::value)
          logp -= log2;
        if (include_summand<propto, T_inv_scale>::value)
          logp += log(lambda_dbl);
        if (include_summand<propto, T_y, T_loc, T_scale, T_inv_scale>::value)
          logp += lambda_dbl
            * (mu_minus_y + 0.5 * lambda_sigma_sq)
            + log(erfc((mu_minus_y + lambda_sigma_sq)
                       / sqrt2_sigma));

        // gradients
        partials_return_t deriv_logerfc
          = neg_two_over_sqrt_pi
          * exp(-square(mu_minus_y_plus_lambda_sigma_sq_over_sqrt2_sigma))
          / erfc(mu_minus_y_plus_lambda_sigma_sq_over_sqrt2_sigma);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n]
            -= lambda_dbl + deriv_logerfc / sqrt2_sigma;
        if (!is_constant_struct<T_loc>::value)
          operands_and_partials.d_x2[n]
            += lambda_dbl + deriv_logerfc / sqrt2_sigma;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x3[n]
            += sigma_dbl * square(lambda_dbl)
            + deriv_logerfc * (-mu_minus_y / sqrt2_sigma_sq
                               + lambda_dbl / sqrt2);
        if (!is_constant_struct<T_inv_scale>::value)
          operands_and_partials.d_x4[n]
            += 1 / lambda_dbl + lambda_sigma_sq
            + mu_minus_y + deriv_logerfc * sigma_dbl / sqrt2;
      }
      return operands_and_partials.value(logp);
    }

    template <typename T_y, typename T_loc, typename T_scale,
              typename T_inv_scale>
    inline
    typename return_type<T_y, T_loc, T_scale, T_inv_scale>::type
    exp_mod_normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
                       const T_inv_scale& lambda) {
      return exp_mod_normal_log<false>(y, mu, sigma, lambda);
    }

  }
}
#endif

