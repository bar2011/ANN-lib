#pragma once

#include <stddef.h>

namespace Math {

// Base matrix class - pure virtual interface, can't be instantiated. only to
// inherit for other classes
template <typename T> class MatrixBase {
public:
  // Get specific item from matrix
  virtual const T &operator[](const size_t row, const size_t col) const = 0;

  virtual void reshape(const size_t rows, const size_t cols) = 0;

  // Getters
  virtual size_t rows() const = 0;
  virtual size_t cols() const = 0;
};
} // namespace Math
