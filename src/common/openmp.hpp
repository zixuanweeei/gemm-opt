
#ifndef COMMON_OPENMP_HPP
#define COMMON_OPENMP_HPP

#include <omp.h>

#include "./common.hpp"

namespace go {

struct omp {
  static omp *get() {
    static omp instance;
    return &instance;
  }

  int get_max_num_threads() {
    const int omp_max_num = omp_get_max_threads();
    return get_int_env("OMP_NUM_THREADS", omp_max_num);
  }

private:
  omp() = default;
};

} // namespace go

#endif
