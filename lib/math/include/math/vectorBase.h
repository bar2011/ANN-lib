#pragma once

#include <stddef.h>
#include <vector>

namespace Math {

// Base vector class - pure virtual interface, can't be instantiated. only to
// inherit for other classes
template <typename T> class VectorBase {
public:
  // Get specific item from matrix
  virtual const T &operator[](size_t index) const = 0;

  // Getters
  virtual size_t size() const = 0;
  virtual const std::vector<T> &data() const = 0;
};
} // namespace Math
