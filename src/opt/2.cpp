
#include <string>

#include "common/common.hpp"
#include "common/parser.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

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
    for (int i = 0; i < M; i += 4) {
      for (int j = 0; j < N; j += 4) {
        float c00 = 0;
        float c01 = 0;
        float c02 = 0;
        float c03 = 0;
        float c10 = 0;
        float c11 = 0;
        float c12 = 0;
        float c13 = 0;
        float c20 = 0;
        float c21 = 0;
        float c22 = 0;
        float c23 = 0;
        float c30 = 0;
        float c31 = 0;
        float c32 = 0;
        float c33 = 0;

        //float * b0 = nullptr;
        for (int k = 0; k < K; k++) {
          float b0 = B[k * N + j];
          float b1 = B[k * N + j + 1];
          float b2 = B[k * N + j + 2];
          float b3 = B[k * N + j + 3];

          float a0 = A[i * K + k];
          float a1 = A[(i + 1) * K + k];
          float a2 = A[(i + 2) * K + k];
          float a3 = A[(i + 3) * K + k];

          c00 += a0 * b0;
          c01 += a0 * b1;
          c02 += a0 * b2;
          c03 += a0 * b3;

          c10 += a1 * b0;
          c11 += a1 * b1;
          c12 += a1 * b2;
          c13 += a1 * b3;

          c20 += a2 * b0;
          c21 += a2 * b1;
          c22 += a2 * b2;
          c23 += a2 * b3;

          c30 += a3 * b0;
          c31 += a3 * b1;
          c32 += a3 * b2;
          c33 += a3 * b3;
        }
        C[i * N + j] = c00;
        C[i * N + j + 1] = c01;
        C[i * N + j + 2] = c02;
        C[i * N + j + 3] = c03;

        C[(i + 1) * N + j] = c10;
        C[(i + 1) * N + j + 1] = c11;
        C[(i + 1) * N + j + 2] = c12;
        C[(i + 1) * N + j + 3] = c13;

        C[(i + 2) * N + j] = c20;
        C[(i + 2) * N + j + 1] = c21;
        C[(i + 2) * N + j + 2] = c22;
        C[(i + 2) * N + j + 3] = c23;

        C[(i + 3) * N + j] = c30;
        C[(i + 3) * N + j + 1] = c31;
        C[(i + 3) * N + j + 2] = c32;
        C[(i + 3) * N + j + 3] = c33;
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
  go::GemmPerf<opt2> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
