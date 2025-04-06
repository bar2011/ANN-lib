#pragma once

#include <stdexcept>
#include <string>

namespace Math {
class Exception : public std::runtime_error {
public:
  explicit Exception(const std::string &function, const std::string &error)
      : std::runtime_error("MathHelpers Exception at " + function + ":\n" + error) {}
};
} // namespace Math
