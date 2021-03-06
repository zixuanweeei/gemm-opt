
#ifndef COMMON_COMMON_HPP
#define COMMON_COMMON_HPP

#include <algorithm>
#include <cstddef>
#include <random>
#include <type_traits>

namespace go {
typedef enum { success = 0, failure = 1 } status_t;
namespace status {
const status_t success = success;
const status_t failure = failure;
} // namespace status

template <typename T, typename = void>
struct is_callable : std::false_type {};

template <typename T>
struct is_callable<T, std::void_t<decltype(std::declval<T>()())>> : std::true_type {};

template <typename T>
void init_random(T *arr, size_t nelems) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> rg(0.f, 0.5f);
  // std::normal_distribution<float> rg(0.f, 1.f);
  // std::uniform_int_distribution<T> rg(0, 10);
  std::generate(arr, arr + nelems, [&] { return rg(gen); });
}

template <typename T>
void init_const(T *arr, size_t nelems, T value) {
  std::fill(arr, arr + nelems, value);
}

} // namespace go

#endif // COMMON_COMMON_HPP
