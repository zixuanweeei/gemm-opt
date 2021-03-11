
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

struct baseline_ikj {
  static constexpr char name[] = "baseline_ikj";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  baseline_ikj(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  static go::status_t kern(
      const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t k = 0; k < K; ++k) {
        float a = A[m * K + k];
        for (size_t n = 0; n < N; ++n) {
          C[m * N + n] += a * B[k * N + n];
        }
      }
    }
    return go::status::success;
  }
};

int main() {
  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<baseline_ikj> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
