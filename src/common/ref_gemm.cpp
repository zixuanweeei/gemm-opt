
#include "ref_gemm.hpp"

namespace go {

void ref_gemm(
    const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float c = 0;
      for (size_t k = 0; k < K; ++k) {
        c += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = c;
    }
  }
}

} // namespace go
