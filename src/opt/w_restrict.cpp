
#include <string>

#include "common/common.hpp"
#include "common/parser.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

struct baseline {
  static constexpr char name[] = "W/ restrict";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  baseline(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  static go::status_t kern(const float *__restrict__ A, const float *__restrict__ B,
      float *__restrict__ C, const size_t M, const size_t N, const size_t K) {
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
          C[m * N + n] += A[m * K + k] * B[k * N + n];
        }
      }
    }
    return go::status::success;
  }
};

#ifndef PERF_ONLY
go_mode_t mode {CORR};
#else
go_mode_t mode {PERF};
#endif

int main(int argc, char **argv) {
  --argc;
  ++argv;
  bool parsed = false;
  for (; argc > 0; --argc, ++argv) {
    parsed |= go::parser::parse_settings(argv[0]);
    if (!parsed) go::parser::catch_unknown_options(argv[0]);
  }

  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<baseline> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
