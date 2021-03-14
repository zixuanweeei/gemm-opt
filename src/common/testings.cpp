
#include <stdio.h>

#include <cmath>
#include <cstddef>
#include <utility>

#include "common.hpp"
#include "ref_gemm.hpp"

namespace go {
namespace testings {

status_t all_close(float *actual, float *desired, const int M, const int N,
    const float rtol = 1.e-6, const float atol = 0.f) {
  std::pair<int, int> maximum_idx;
  float maximum_err = 0.f;
  float tol = 0.f;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      const float a = actual[m * N + n];
      const float b = desired[m * N + n];
      // std::cout << "actual = " << a << ", desired = " << b << std::endl;
      const float err = std::fabs(a - b);
      if (err > maximum_err) {
        maximum_idx = {m, n};
        maximum_err = err;
        tol = atol + rtol * std::fabs(b);
      }
    }
  }
  if (maximum_err > tol) {
    const int m = maximum_idx.first;
    const int n = maximum_idx.second;
    printf("Error! Matrix[%d, %d]=%.8f, ref=%.8f, error = %.8f, error term is > %E\n", m, n,
        actual[m * N + n], desired[m * N + n], maximum_err, tol);
    fflush(stdout);
    return status::failure;
  }
  return status::success;
}

status_t verify_mm(
    const int M, const int N, const int K, const float *A, const float *B, float *C) {
  std::vector<float> cval(M * K);

  for (int i = 0; i < M * K; i++) {
    cval[i] = 0.0f;
  }

  ref_gemm(A, B, cval.data(), M, N, K);

  // check for correctness
  status_t status = all_close(C, cval.data(), M, N, 1.e-5);

  return status;
}

} // namespace testings
} // namespace go
