#pragma once

#include <memory>
#include <optional>
#include <stddef.h>
#include <vector>

namespace Math {

template <typename T> class Vector;

template <typename T> class VectorView;

template <typename T> class Matrix;

template <typename T> class MatrixView;

// Base matrix class - pure virtual interface, can't be instantiated. only to
// inherit for other classes
template <typename T> class MatrixBase {
public:
  // Single item access - NO BOUNDS CHECKING
  virtual const T &operator[](const size_t row, const size_t col) const = 0;
  // Single item access - WITH BOUNDS CHECKING
  virtual const T &at(const size_t row, const size_t col) const = 0;

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

  virtual const std::vector<T> &data() const = 0;

  virtual std::unique_ptr<Math::VectorView<T>> asVector() = 0;

  // Returns a view of the entire matrix
  virtual std::shared_ptr<Math::MatrixView<T>> view() const = 0;

  // Returns a view of a range of rows from the matrix.
  // Includes rows in the range [startRow, endRow), i.e., startRow is inclusive,
  // endRow is exclusive. The view includes all columns in each row.
  // Throws if endRow > row count or startRow >= endRow.
  virtual std::shared_ptr<MatrixView<T>> view(size_t startRow,
                                              size_t endRow) const = 0;

  // Transposes the matrix. Returns the transposed one.
  // Note: the returned matrix has complete ownership on its values
  virtual std::shared_ptr<Matrix<T>>
  transpose(size_t chunkSize = 4,
            std::optional<bool> parallelize = std::nullopt) const = 0;
};
} // namespace Math
