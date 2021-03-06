
#ifndef COMMON_TESTINGS_HPP
#define COMMON_TESTINGS_HPP

#include <cstddef>

#include "common.hpp"

namespace go {
namespace testings {

status_t verify_mm(const int M, const int N, const int K, const float *A, const float *B, float *C);

} // namespace testings
} // namespace go

#endif // COMMON_TESTINGS_HPP
