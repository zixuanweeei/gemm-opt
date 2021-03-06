
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

struct baseline {
  static constexpr char name[] = "baseline";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  baseline(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  static go::status_t kern(
      const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
    go::ref_gemm(A, B, C, M, N, K);
    return go::status::success;
  }
};

int main() {
  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<baseline> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
