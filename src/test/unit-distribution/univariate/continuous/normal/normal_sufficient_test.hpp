// Arguments: Doubles, Doubles, Ints, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionNormalSufficient : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(5);

    //observed values: 0, 0
    param[0] = 0;           // y_bar
    param[1] = 0;           // s_squared
    param[2] = 2;           // n_obs
    param[3] = 0;           // mu
    param[4] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.837877); // expected log_prob
    
    //observed values: 0, 1
    param[0] = 0.5;           // y_bar
    param[1] = 0.5;           // s_squared
    param[2] = 2;           // n_obs
    param[3] = 0;           // mu
    param[4] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.337877); // expected log_prob

    
    //observed values: 0, 2
    param[0] = 1;           // y_bar
    param[1] = 2;           // s_squared
    param[2] = 2;           // n_obs
    param[3] = 0;           // mu
    param[4] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.837877); // expected log_prob

    //observed values: 1, 2
    param[0] = 1.5;           // y_bar
    param[1] = 0.5;           // s_squared
    param[2] = 2;           // n_obs
    param[3] = 0;           // mu
    param[4] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-4.337877); // expected log_prob
  }

  void invalid_values(vector<size_t>& index,
          vector<double>& value) {
    // y

    // mu
    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(4U);
    value.push_back(0.0);

    index.push_back(4U);
    value.push_back(-1.0);

    index.push_back(4U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_s, typename T_n,
      typename T_loc, typename T_scale, typename T5,
      typename T6, typename T7, typename T8,
      typename T9>
  typename stan::return_type<T_y, T_s, T_n, T_loc, T_scale>::type
  log_prob(const T_y& y_bar, const T_s& s_squared, const T_n& n_obs,
     const T_loc& mu, const T_scale& sigma,
     const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_sufficient_log(y_bar, s_squared, n_obs, mu, sigma);
  }

  template <bool propto,
      typename T_y, typename T_s, typename T_n,
      typename T_loc, typename T_scale, typename T5,
      typename T6, typename T7, typename T8,
      typename T9>
  typename stan::return_type<T_y, T_s, T_n, T_loc, T_scale>::type
  log_prob(const T_y& y_bar, const T_s& s_squared, const T_n& n_obs,
     const T_loc& mu, const T_scale& sigma,
     const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_sufficient_log<propto>(y_bar, s_squared, n_obs, mu, sigma);
  }


  template <typename T_y, typename T_s, typename T_n,
      typename T_loc, typename T_scale, typename T5,
      typename T6, typename T7, typename T8,
      typename T9>
  var log_prob_function(const T_y& y_bar, const T_s& s_squared, const T_n& n_obs,
     const T_loc& mu, const T_scale& sigma,
     const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::pi;
    using stan::math::square;
    var lp(0.0);
    if (include_summand<true,T_scale>::value)
      lp -= n_obs * log(sigma);
    
      lp -= (s_squared + n_obs * pow(y_bar - mu, 2)) / (2 * pow(sigma, 2));
      
    if (include_summand<true>::value)
      lp -= log(sqrt(2.0 * pi()));
    return lp;
  }
};
