#ifndef STAN_MATH_PRIM_MAT_FUN_CROSSPROD_HPP
#define STAN_MATH_PRIM_MAT_FUN_CROSSPROD_HPP

#include <stan/math/prim/mat/fun/tcrossprod.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the result of pre-multiplying a matrix by its own
     * transpose.
     *
     * @param M Matrix argument.
     * @return Transpose of argument times itself.
     */
    inline Eigen::MatrixXd crossprod(const Eigen::MatrixXd& M) {
      return tcrossprod(static_cast<Eigen::MatrixXd>(M.transpose()));
    }

  }
}
#endif
