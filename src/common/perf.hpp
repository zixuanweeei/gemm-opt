
#ifndef COMMON_PERF_HPP
#define COMMON_PERF_HPP

#include <algorithm>
#include <chrono>
#include <iostream>

#include "common.hpp"
#include "testings.hpp"

namespace go {

template <typename T>
struct GemmPerf {
  static_assert(is_callable<T>::value, "Type is not callable.");

  static constexpr size_t warmup = 10;
  static constexpr size_t dryrun = 100;

  size_t M, N, K;
  size_t ops;
  size_t gflops;
  double us;

  std::vector<float> containerA;
  std::vector<float> containerB;
  std::vector<float> containerC;

  float *A;
  float *B;
  float *C;

  bool is_initialized = false;

  T gemm;

  GemmPerf() = default;
  GemmPerf(const size_t M, const size_t N, const size_t K)
    : M(M)
    , N(N)
    , K(K)
    , ops(M * N * 2 * K)
    , gflops(0)
    , us(0)
    , containerA(M * K, 0.f)
    , containerB(K * N, 0.f)
    , containerC(M * N, 0.f)
    , A(containerA.data())
    , B(containerB.data())
    , C(containerC.data())
    , is_initialized(true)
    , gemm {A, B, C, M, N, K} {
    init_random(A, M * K);
    init_random(B, K * N);
    init_const(C, M * N, 0.f);
  };

  // finish correctness verification and performance measurement
  status_t finalize() {
    std::chrono::duration<double, std::micro> durations(0);
    for (size_t iter = 0; iter < warmup + dryrun; ++iter) {
      std::fill(C, C + M * N, 0.f);
      auto start = std::chrono::high_resolution_clock::now();
      gemm();
      std::chrono::duration<double, std::micro> delta
          = std::chrono::high_resolution_clock::now() - start;
      if (iter > warmup) durations += delta;
    }
    us = durations.count() / dryrun;

    status_t ret = status::success;
    if (mode & CORR) ret = testings::verify_mm(M, N, K, A, B, C);

    return ret;
  }

  // report gemm perf
  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const GemmPerf<U> &gp);
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const GemmPerf<T> &gp) {
  double us = gp.us;
  size_t ops = gp.ops;
  double gflops = ops * 1.e-9 / us * 1.e6;

  char buffer[512] = {'\0'};
  sprintf(buffer, "Name: %s, Dim: %ldx%ldx%ld, ms/r: %.3f, Glops: %.4f\n", T::name, gp.M, gp.N,
      gp.K, us / 1000.f, gflops);
  os << buffer;

  return os;
}

} // namespace go

#endif // COMMON_PERF_HPP
