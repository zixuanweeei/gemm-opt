
#include <string>

#include "common/common.hpp"
#include "common/parser.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

#include "emmintrin.h"
#include "immintrin.h"

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define TILE_SIZE 8

struct opt6 {
  static constexpr char name[] = "opt6-vec8";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  opt6(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  template <int block_factor_m = 32, int block_factor_n = 32, int tile = 32>
  struct Kern {
    static constexpr int blk_size_m = block_factor_m;
    static constexpr int blk_size_n = block_factor_n;
    static constexpr int SIMD_width = sizeof(__m256) / sizeof(float);
    static constexpr int blk_SIMD_n = blk_size_n / SIMD_width;
    static constexpr int tile_size = tile;

    go::status_t operator()(
        const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
      return kernel(A, B, C, M, N, K);
    }

    static go::status_t kernel(
        const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
      const int m_outer = M / blk_size_m;
      const int n_outer = N / blk_size_n;
      const int k_outer = K / tile_size;

      for (int om = 0; om < m_outer; ++om) {
        for (int on = 0; on < n_outer; ++on) {
          int gm = om * blk_size_m;
          int gn = on * blk_size_n;
          __m256 c[blk_size_m][blk_SIMD_n];
          for (int blk = 0; blk < blk_size_m; ++blk) {
            for (int lane = 0; lane < blk_SIMD_n; ++lane) {
              c[blk][lane] = _mm256_setzero_ps();
            }
          }
          for (int ok = 0; ok < k_outer; ++ok) {
            int gk = ok * tile_size;
            for (int im = 0; im < blk_size_m; ++im) {
              for (int ik = 0; ik < tile_size; ++ik) {
                for (int lane = 0; lane < blk_SIMD_n; ++lane) {
                  __m256 vec_b = _mm256_loadu_ps(B + (gk + ik) * N + gn + lane * SIMD_width);
                  __m256 vec_a_dup = _mm256_broadcast_ss(A + (gm + im) * K + gk + ik);
                  c[im][lane] = _mm256_fmadd_ps(vec_a_dup, vec_b, c[im][lane]);
                }
              }
            }
          }
          for (int m = 0; m < blk_size_m; ++m) {
            for (int lane = 0; lane < blk_SIMD_n; ++lane) {
              _mm256_storeu_ps(C + (gm + m) * N + gn + lane * SIMD_width, c[m][lane]);
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
  go::GemmPerf<opt6> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
