
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define TILE_SIZE 16

struct opt3 {
  static constexpr char name[] = "opt3";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  opt3(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  template <int block_factor_m = 32, int block_factor_n = 32, int tile = 32>
  struct Kern {
    static constexpr int blk_size_m = block_factor_m;
    static constexpr int blk_size_n = block_factor_n;
    static constexpr int tile_size = tile;

    go::status_t operator()(
        const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
      return kernel(A, B, C, M, N, K);
    }

    static go::status_t kernel(
        const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
      const size_t m_outer = M / blk_size_m;
      const size_t n_outer = N / blk_size_n;
      const size_t k_outer = K / tile_size;

      for (size_t om = 0; om < m_outer; ++om) {
        for (size_t on = 0; on < n_outer; ++on) {
          size_t gm = om * blk_size_m;
          size_t gn = on * blk_size_n;
          float c[blk_size_m][blk_size_n] = {0};
          for (size_t ok = 0; ok < k_outer; ++ok) {
            size_t gk = ok * tile_size;
            for (int ik = 0; ik < tile_size; ++ik) {
              for (int im = 0; im < blk_size_m; ++im) {
                for (int in = 0; in < blk_size_n; ++in) {
                  c[im][in] += A[(gm + im) * K + gk + ik] * B[(gk + ik) * N + gn + in];
                }
              }
            }
          }
          for (int m = 0; m < blk_size_m; ++m) {
            for (int n = 0; n < blk_size_n; ++n) {
              C[(gm + m) * N + gn + n] = c[m][n];
            }
          }
        }
      }

      return go::success;
    }
  };

  static go::status_t kern(
      const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
    Kern<BLOCK_SIZE_M, BLOCK_SIZE_N, TILE_SIZE> k;
    return k(A, B, C, M, N, K);
  }
};

int main() {
  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<opt3> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
