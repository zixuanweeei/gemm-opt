
#include "./parser.hpp"
#include "./common.hpp"

namespace go {
namespace parser {

static bool parse_mode(const char *str, const std::string &option_name = "mode") {
  return parse_single_value_option(mode, PERF, str2mode, str, option_name);
}

bool parse_settings(const char *str) {
  return parse_mode(str);
}

void catch_unknown_options(const char *str) {
  const std::string pattern = "--";
  if (pattern.find(str, 0, pattern.size()) != std::string::npos) {
    fprintf(stderr, "ERROR: unknown option: `%s`, existing...\n", str);
    exit(2);
  }
}

} // namespace parser
} // namespace go
