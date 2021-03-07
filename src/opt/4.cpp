
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

#include "immintrin.h"
#include "xmmintrin.h"

struct opt2 {
  static constexpr char name[] = "opt2";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  opt2(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  static go::status_t kern(
      const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
    __m128 t0, t1;
    __m128 c0, c1, c2, c3;
    for (int i = 0; i < M; i += 4) {
      for (int j = 0; j < N; j += 4) {
        c0 = _mm_setzero_ps();
        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
        for (int k = 0; k < K; k++) {
          t1 = _mm_load_ps(&B[k * N + j]);

          t0 = _mm_load_ps1(&A[i * K + k]);  // Load and duplicate
          c0 = _mm_fmadd_ps(t0, t1, c0);

          t0 = _mm_load_ps1(&A[(i + 1) * K + k]);
          c1 = _mm_fmadd_ps(t0, t1, c1);

          t0 = _mm_load_ps1(&A[(i + 2) * K + k]);
          c2 = _mm_fmadd_ps(t0, t1, c2);

          t0 = _mm_load_ps1(&A[(i + 3) * K + k]);
          c3 = _mm_fmadd_ps(t0, t1, c3);
        }
        _mm_store_ps(&C[i * N + j], c0);
        _mm_store_ps(&C[(i + 1) * N + j], c1);
        _mm_store_ps(&C[(i + 2) * N + j], c2);
        _mm_store_ps(&C[(i + 3) * N + j], c3);
      }
    }
    return go::status::success;
  }
};

int main() {
  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<opt2> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
