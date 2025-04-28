#pragma once

#include "math/vector.h"
#include <memory>
#include <stddef.h>

namespace Math {

// Base matrix class - pure virtual interface, can't be instantiated. only to
// inherit for other classes
template <typename T> class MatrixBase {
public:
  // Get specific item from matrix
  virtual const T &operator[](const size_t row, const size_t col) const = 0;

  virtual MatrixBase &reshape(const size_t rows, const size_t cols) = 0;

  // Getters
  virtual size_t rows() const = 0;
  virtual size_t cols() const = 0;

  // Returns the index of the biggest value in the matrix
  // format: (row, col)
  virtual std::pair<size_t, size_t> argmax() const = 0;

  // Returns a vector containing the index of the biggest value in each row
  virtual std::unique_ptr<Math::Vector<size_t>> argmaxRow() const = 0;

  // Returns a vector containing the index of the biggest value in each column
  virtual std::unique_ptr<Math::Vector<size_t>> argmaxCol() const = 0;
};
} // namespace Math
