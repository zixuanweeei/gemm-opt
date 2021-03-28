
#ifndef COMMON_PARSER_HPP
#define COMMON_PARSER_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace go {
namespace parser {

static inline std::string get_pattern(const std::string &option_name) {
  return std::string("--") + option_name + std::string("=");
}

template <typename T, typename F>
static bool parse_single_value_option(
    T &val, const T &def_val, F process_func, const char *str, const std::string &option_name) {
  const std::string pattern = get_pattern(option_name);
  if (pattern.find(str, 0, pattern.size()) == std::string::npos) return false;
  str = str + pattern.size();
  if (*str == '\0') return val = def_val, true;
  return val = process_func(str), true;
}

bool parse_settings(const char *str);
void catch_unknown_options(const char *str);

} // namespace parser
} // namespace go

#endif
