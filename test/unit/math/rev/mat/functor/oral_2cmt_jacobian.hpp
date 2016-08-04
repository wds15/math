#ifndef TEST_UNIT_MATH_REV_MAT_FUNCTOR_ORAL_2CMT_JACOBIAN_HPP
#define TEST_UNIT_MATH_REV_MAT_FUNCTOR_ORAL_2CMT_JACOBIAN_HPP

#include <stan/math/rev/mat/functor/ode_system.hpp>
#include <test/unit/math/rev/mat/functor/oral_2cmt.hpp>

// define an analytic jacobian for the 2cmt oral ODE for Stan's CVODES
// & RK45 solver by partial template specialisation


namespace stan {
  namespace math {

    // todo: idealy, we derive the specialised ode_model from the
    // general one which gives possibly syntax problem which is why we
    // chose to use a typedef, maybe we need another helper... not
    // sure.
    template<>
    struct ode_system<oral_2cmt_ode_fun> {
      typedef oral_2cmt_ode_fun F;
      const F& f_;
      const std::vector<double> theta_;
      const std::vector<double>& x_;
      const std::vector<int>& x_int_;
      std::ostream* msgs_;

      ode_system(const F& f,
                 const std::vector<double> theta,
                 const std::vector<double>& x,
                 const std::vector<int>& x_int,
                 std::ostream* msgs)
	: f_(f),
	  theta_(theta),
	  x_(x),
	  x_int_(x_int),
	  msgs_(msgs)
      {}

      inline void operator()(const double t,
                             const std::vector<double>& y,
                             std::vector<double>& dy_dt) const {
	dy_dt = f_(t, y, theta_, x_, x_int_, msgs_);
      }

      template <typename Derived1, typename Derived2>
      void
      jacobian(const double t,
               const std::vector<double>& y,
               Eigen::MatrixBase<Derived1>& fy,
               Eigen::MatrixBase<Derived2>& Jy
               ) const {
	using Eigen::VectorXd;
	using std::vector;
	using std::pow;

	const vector<double> f = f_(t, y, theta_, x_, x_int_, msgs_);
	fy = VectorXd::Map(&f[0], f.size());

	const double ka = theta_[0];
	const double k10 = theta_[1];
	const double k12 = theta_[2];
	const double k21 = theta_[3];

	Jy.setZero();
        // column major ordering
	Eigen::Map<Eigen::VectorXd> J = Eigen::Map<Eigen::VectorXd>(&Jy(0,0), Jy.cols()*Jy.rows());

	 J[0] = -ka;
 	 J[1] = ka;
 	 J[2] = 0;
 	 J[3] = 0;
 	 J[4] = -(k10+k12);
 	 J[5] = k12;
 	 J[6] = 0;
 	 J[7] = k21;
 	 J[8] = -k21;
      }

      template <typename Derived1, typename Derived2, typename Derived3>
      void
      jacobian(const double t,
               const std::vector<double>& y,
               Eigen::MatrixBase<Derived1>& fy,
               Eigen::MatrixBase<Derived2>& Jy,
               Eigen::MatrixBase<Derived3>& Jtheta
               ) const {
	using Eigen::VectorXd;
	using std::vector;
	using std::pow;

	jacobian(t, y, fy, Jy);

	const double a = y[0];
	const double c = y[1];
	const double p = y[2];

	//const vector<double>& parms = theta_;

	Jtheta.setZero();

        // column major ordering
	Eigen::Map<Eigen::VectorXd> J = Eigen::Map<Eigen::VectorXd>(&Jtheta(0,0), Jtheta.cols()*Jtheta.rows());

	J[0] = -a;
	J[1] = a;
	J[2] = 0;
	J[3] = 0;
	J[4] = -c;
	J[5] = 0;
	J[6] = 0;
	J[7] = -c;
	J[8] = c;
	J[9] = 0;
	J[10] = p;
	J[11] = -p;
      }

    };

  }
}

#endif
