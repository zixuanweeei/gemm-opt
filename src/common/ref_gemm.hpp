
#ifndef COMMON_REF_GEMM_HPP
#define COMMON_REF_GEMM_HPP

#include <cstddef>

#include <inttypes.h>

namespace go {

void ref_gemm(
    const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K);

} // namespace go

#endif // COMMON_REF_GEMM_HPP
