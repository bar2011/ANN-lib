#pragma once

#include <stdexcept>
#include <string>

namespace Loaders {
class Exception : public std::runtime_error {
public:
  explicit Exception(const std::string &function, const std::string &error)
      : std::runtime_error("Loader Exception at " + function + ":\n" + error) {}
};
} // namespace Loaders
