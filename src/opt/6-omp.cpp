
#include <omp.h>
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

#include "immintrin.h"
#include "xmmintrin.h"

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define TILE_SIZE 16

struct opt6 {
  static constexpr char name[] = "opt6";

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
    static constexpr int SIMD_width = sizeof(__m128) / sizeof(float);
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

#pragma omp parallel for num_threads(4) collapse(2)
      for (int om = 0; om < m_outer; ++om) {
        for (int on = 0; on < n_outer; ++on) {
          int gm = om * blk_size_m;
          int gn = on * blk_size_n;
          __m128 c[blk_size_m][blk_SIMD_n];
          for (int blk = 0; blk < blk_size_m; ++blk) {
            for (int lane = 0; lane < blk_SIMD_n; ++lane) {
              c[blk][lane] = _mm_setzero_ps();
            }
          }
          for (int ok = 0; ok < k_outer; ++ok) {
            int gk = ok * tile_size;
            for (int im = 0; im < blk_size_m; ++im) {
              for (int ik = 0; ik < tile_size; ++ik) {
                for (int lane = 0; lane < blk_SIMD_n; ++lane) {
                  __m128 vec_b = _mm_load_ps(B + (gk + ik) * N + gn + lane * SIMD_width);
                  __m128 vec_a_dup = _mm_load_ps1(A + (gm + im) * K + gk + ik);
                  c[im][lane] = _mm_fmadd_ps(vec_a_dup, vec_b, c[im][lane]);
                }
              }
            }
          }
          for (int m = 0; m < blk_size_m; ++m) {
            for (int lane = 0; lane < blk_SIMD_n; ++lane) {
              _mm_store_ps(C + (gm + m) * N + gn + lane * SIMD_width, c[m][lane]);
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
  go::GemmPerf<opt6> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
