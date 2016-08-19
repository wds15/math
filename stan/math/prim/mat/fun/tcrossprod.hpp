#ifndef STAN_MATH_PRIM_MAT_FUN_TCROSSPROD_HPP
#define STAN_MATH_PRIM_MAT_FUN_TCROSSPROD_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the result of post-multiplying a matrix by its own
     * transpose.
     *
     * @param[in] M Matrix to multiply.
     * @return M times its transpose.
     */
    inline Eigen::MatrixXd tcrossprod(const Eigen::MatrixXd& M) {
        if (M.rows() == 0)
          return Eigen::MatrixXd(0, 0);
        if (M.rows() == 1)
          return M * M.transpose();
        Eigen::MatrixXd a(M.rows(), M.rows());
        return a.setZero()
          .selfadjointView<Eigen::Upper>()
          .rankUpdate(M);
    }

  }
}
#endif
