
#include <cassert>
#include <stdexcept>
#include <string.h>

#include "./common.hpp"

const char *mode2str(go_mode_t mode) {
  const char *modes[] = {"MODE_UNDEF", "PERF", "CORR"};
  assert((int)mode < sizeof(modes) / sizeof(*modes));
  return modes[(int)mode];
}

go_mode_t str2mode(const char *str) {
  go_mode_t mode = MODE_UNDEF;
  if (strchr(str, 'p') || strchr(str, 'P')) mode = (go_mode_t)((int)mode | (int)PERF);
  if (strchr(str, 'c') || strchr(str, 'C')) mode = (go_mode_t)((int)mode | (int)CORR);
  if (mode == MODE_UNDEF) throw std::invalid_argument("Undefined mode is detected.");
  return mode;
}
