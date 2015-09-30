#ifndef STAN_MATH_REV_CORE_SET_ZERO_ALL_ADJOINTS_NESTED_HPP
#define STAN_MATH_REV_CORE_SET_ZERO_ALL_ADJOINTS_NESTED_HPP

#include <stan/math/rev/core/chainable.hpp>
#include <stan/math/rev/core/chainable_alloc.hpp>
#include <stan/math/rev/core/chainablestack.hpp>

namespace stan {
  namespace math {

    /**
     * Reset all adjoint values in the top nested portion of the stack
     * to zero.
     */
    static void set_zero_all_adjoints_nested() {
      if (empty_nested())
        throw std::logic_error("empty_nested() must be false before calling"
                               " set_zero_all_adjoints_nested()");
      size_t start1 = ChainableStack::nested_var_stack_sizes_.back();
      for (size_t i = start1 - 1; i < ChainableStack::var_stack_.size(); ++i)
        ChainableStack::var_stack_[i]->set_zero_adjoint();


      size_t start2 = ChainableStack::nested_var_nochain_stack_sizes_.back();
      for (size_t i = start2 - 1;
           i < ChainableStack::var_nochain_stack_.size(); ++i)
        ChainableStack::var_nochain_stack_[i]->set_zero_adjoint();
    }

  }
}
#endif