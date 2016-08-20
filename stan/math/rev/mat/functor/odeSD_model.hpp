#pragma once

#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <math.h>

#include <stan/math/rev/mat/functor/ode_system.hpp>

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>

#include <boost/type_traits/is_same.hpp>

#include <odeSD/SensitivityModel>

namespace stan {

  namespace math {

    // glues f and f' together such that we can use it in ode_model
    template<typename F, typename FP>
    class odeSD_adapter {
      const F& f_;
      const FP& fp_;
      
    public:
      odeSD_adapter(const F& f, const FP& fp) : f_(f), fp_(fp) {}
      
      template <typename T0, typename T1, typename T2>
      inline
      std::vector<typename stan::return_type<T1,T2>::type>
      operator()(const T0& t_in, // initial time
		 const std::vector<T1>& y, //initial positions
		 const std::vector<T2>& theta, // parameters
		 const std::vector<double>& sx, // double data
		 const std::vector<int>& sx_int,  // integer data
		 std::ostream* msgs) const {
	std::vector<typename stan::return_type<T1,T2>::type> f = f_(t_in, y, theta, sx, sx_int, msgs);
	const std::vector<typename stan::return_type<T1,T2>::type> fPrime = fp_(t_in, y, f, theta, sx, sx_int, msgs);
	f.reserve(f.size() + fPrime.size());
	f.insert(f.end(), fPrime.begin(), fPrime.end());
	return(f);
      }
    };

    
    template<typename F, typename FP, typename T1, typename T2>
    struct odeSD_model :
      public odeSD::SensitivityModel {

      const size_t N_;
      const size_t M_;
      const size_t S_;
      odeSD::SensitivityState x0_;
      const size_t size_;
      const size_t theta_var_ind_;
      std::vector<var> vars_;

      // returned solution type
      typedef std::vector<typename stan::return_type<T1,T2>::type> state_t;
      typedef boost::is_same<T1, var> initial_var;
      typedef boost::is_same<T2, var> theta_var;
      
      typedef odeSD_adapter<F, FP> combined_f;
      ode_system<combined_f> ode_system_;

      // structure which holds the final solution
      //std::vector<state_t> x_var;
      std::vector<std::vector<double> > x_var;

      odeSD_model(const F& f, const FP& fp,
		  const std::vector<double>& x0,
		  const std::vector<double>& theta,
		  const std::vector<double>& sx,
		  const std::vector<int>& sx_int,
		  std::ostream* msgs,
		  const std::vector<double>& ts)
	: N_(x0.size()),
	  M_(theta.size()),
	  S_((initial_var::value ? N_ : 0) + (theta_var::value ? M_ : 0)),
	  x0_(N_, S_),
	  size_(N_ * (S_+1)),
	  theta_var_ind_(initial_var::value ? N_ : 0),
	  vars_(S_),
	  ode_system_(combined_f(f, fp), theta, sx, sx_int, msgs),
	  x_var(ts.size() - 1, std::vector<double>(size_))
	  //x_var(ts.size(), state_t(N_))
      {
	using Eigen::VectorXd;
	x0_.x = VectorXd::Map(&x0[0], N_);
	x0_.s.setZero();
	
	if(initial_var::value)
	  x0_.s.leftCols(N_).setIdentity();
	
	//parameters = VectorXd::Map(&theta[0], M_);

	tspan = VectorXd::Map(&ts[0], ts.size());

	//states = Eigen::MatrixXd(N_, tspan.rows());
	//sensitivities = Eigen::MatrixXd(x0_.nStates * x0_.nParameters, tspan.rows());

	if(initial_var::value) {
	  for(size_t n=0; n < N_; n++) {
	    vars_[n] = x0[n];
	  }
	}
	if(theta_var::value) {
	  for(size_t m=0; m < M_; m++) {
	    vars_[theta_var_ind_ + m] = theta[m];
	  }
	}
      }

      virtual odeSD::SensitivityState& getInitialState()
      {
	return x0_;
      }

      // time, current state, f, f'
      virtual void calculateRHS(odeSD::State& state)
      {
	const std::vector<double> xv(&state.x(0), &state.x(0) + N_);
	
	std::vector<double> fc(2*N_);
	ode_system_(state.t, xv, fc);

	state.f      = Eigen::VectorXd::Map(&fc[0 ], N_);
	state.fPrime = Eigen::VectorXd::Map(&fc[N_], N_);
      }

      // ignore those intermediate states
      virtual void observeIntermediateState(Eigen::DenseIndex stepIndex, const odeSD::State& state) {}
      virtual void observeIntermediateSensitivityState(Eigen::DenseIndex stepIndex, const odeSD::SensitivityState& state) {}
      virtual void observeIntegrationEnd() {}

      virtual void observeNewState(Eigen::DenseIndex timeIndex, const odeSD::State& state) {
	size_t n = timeIndex; // the initial state is dropped
	if(n > 0) {
	  //Eigen::VectorXd::Map(&x_var[n][0], N_) = state.x;
	  /**/
	  for(size_t i = 0; i < N_; i++) {
	    x_var[n-1][i] = state.x(i);
	  }
	}
	/**/
      }
      
      virtual void observeNewSensitivityState(Eigen::DenseIndex timeIndex, const odeSD::SensitivityState& state) {
	size_t n = timeIndex; // the initial state is dropped
	if(n > 0)
	  record_sens_state(n-1, state, state_t());
      }

      // in case the sensitiviy solver is used, but no sensitivities
      // are calculated at all, we can discard this call since the
      // state has already been recorde by observe new state
      void record_sens_state(const size_t n, const odeSD::SensitivityState& state, const std::vector<double>& state_) {
      }
      
      void record_sens_state(const size_t n, const odeSD::SensitivityState& state, const std::vector<var>& state_) {
	//Eigen::VectorXd::Map(&x_var[n][0], N_) = state.x;
	
	// use the fact that Eigen stores by column order which is the
	// same as the usual order in x_var is
	//std::copy(state.s.data(), state.s.data() + S_ * N_, &x_var[n][N_]);
	/**/
	for(size_t i = 0; i < N_; i++) {
	  x_var[n][i] = state.x(i);
	  
	  for(size_t s = 0; s < S_; s++) {
	    for(size_t j = 0; j < N_; j++) {
	      x_var[n][N_ + s * N_ + j] = state.s(j,s);
	    }
	  }
	}
	/**/
      }
  
      virtual void calculateJac(odeSD::JacobianState& jacState, odeSD::State& state)
      {
	Eigen::VectorXd fc(2*N_);
	Eigen::MatrixXd Jc(2*N_, N_);

	std::vector<double> xv(N_);
	Eigen::VectorXd::Map(&xv[0], N_) = state.x;

	ode_system_.jacobian(state.t, xv, fc, Jc);

	jacState.dfdx      = Jc.topRows(N_);
	jacState.dfPrimedx = Jc.bottomRows(N_);
      }

      virtual void calculateJacAndSensitivities(odeSD::JacobianState& jacState, odeSD::SensitivityState& state)
      {
	
	Eigen::VectorXd  fc(2*N_);
	Eigen::MatrixXd  Jc(2*N_, N_);
	Eigen::MatrixXd JPc(2*N_, M_);

	const std::vector<double> xv(&state.x(0), &state.x(0) + N_);

	if(theta_var::value) {
	  ode_system_.jacobian(state.t, xv, fc, Jc, JPc);
	} else {
	  ode_system_.jacobian(state.t, xv, fc, Jc);
	}

	jacState.dfdx      = Jc.topRows(N_);
	jacState.dfPrimedx = Jc.bottomRows(N_);

	if(initial_var::value) {
	  state.dfdp.leftCols(N_).setZero();
	  state.dfPrimedp.leftCols(N_).setZero();
	}
	if(theta_var::value) {
	  state.dfdp.rightCols(M_)      = JPc.topRows(N_);
	  state.dfPrimedp.rightCols(M_) = JPc.bottomRows(N_);
	}
      }

      size_t size() const {
        return size_;
      }
      
    };
    
  }  // ns math
}  // ns stan


