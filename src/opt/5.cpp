
#include <string>

#include "common/common.hpp"
#include "common/perf.hpp"
#include "common/ref_gemm.hpp"

#include "immintrin.h"
#include "xmmintrin.h"

#define BLOCK_SIZE 4
#define TILE_SIZE 8

struct opt5 {
  static constexpr char name[] = "opt5";

  size_t M;
  size_t N;
  size_t K;

  float *A;
  float *B;
  float *C;

  opt5(float *A, float *B, float *C, const size_t M, const size_t N, const size_t K)
    : M(M), N(N), K(K), A(A), B(B), C(C) {}

  go::status_t operator()() { return kern(A, B, C, M, N, K); }

  template <int block_factor = 32, int tile = 32>
  struct Kern {
    static constexpr int blk_size = block_factor;
    static constexpr int iblk_size = block_factor;
    static constexpr int tile_size = tile;

    go::status_t operator()(
        const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
      return kernel(A, B, C, M, N, K);
    }

    static go::status_t kernel(
        const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
      const size_t m_outer = M / blk_size;
      const size_t n_outer = N / blk_size;
      const size_t k_outer = K / tile_size;

      for (size_t om = 0; om < m_outer; ++om) {
        for (size_t on = 0; on < n_outer; ++on) {
          size_t gm = om * blk_size;
          size_t gn = on * blk_size;
          __m128 c[blk_size];
#pragma GCC ivdep
          for (size_t blk = 0; blk < blk_size; ++blk) {
            c[blk] = _mm_setzero_ps();
          }
          for (size_t ok = 0; ok < k_outer; ++ok) {
            size_t gk = ok * tile_size;
#pragma GCC ivdep
            for (int ik = 0; ik < tile_size; ++ik) {
              float a[blk_size] = {0};
#pragma GCC ivdep
              for (int blk = 0; blk < blk_size; ++blk) {
                a[blk] = A[(gm + blk) * K + gk + ik];
              }
              __m128 vec_b = _mm_load_ps(B + (gk + ik) * N + gn);
#pragma GCC ivdep
              for (int im = 0; im < blk_size; ++im) {
                __m128 vec_a_dup = _mm_load_ps1(a + im);
                c[im] = _mm_fmadd_ps(vec_a_dup, vec_b, c[im]);
              }
            }
          }
#pragma GCC ivdep
          for (int m = 0; m < blk_size; ++m) {
            _mm_store_ps(C + (gm + m) * N + gn, c[m]);
          }
        }
      }

      return go::success;
    }
  };

  static go::status_t kern(
      const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K) {
    Kern<BLOCK_SIZE, TILE_SIZE> k;
    return k(A, B, C, M, N, K);
  }
};

int main() {
  const size_t M = 1280;
  const size_t N = 1280;
  const size_t K = 1280;
  go::GemmPerf<opt5> perf(M, N, K);
  perf.finalize();
  std::cout << perf;

  return 0;
}
