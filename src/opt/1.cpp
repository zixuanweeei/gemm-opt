
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

struct opt1 {
  static constexpr char name[] = "opt1";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  opt1(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  static go::status_t kern(
      const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j += 4) {
        float c0 = 0;
        float c1 = 0;
        float c2 = 0;
        float c3 = 0;
        for (int k = 0; k < K; k++) {
          float a = A[i * K + k];
          c0 += a * B[k * N + j];
          c1 += a * B[k * N + j + 1];
          c2 += a * B[k * N + j + 2];
          c3 += a * B[k * N + j + 3];
        }
        C[i * N + j] = c0;
        C[i * N + j + 1] = c1;
        C[i * N + j + 2] = c2;
        C[i * N + j + 3] = c3;
      }
    }
    return go::status::success;
  }
};

int main() {
  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<opt1> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
