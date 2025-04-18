#pragma once

#include "matrixBase.h"
#include <vector>

namespace Math {

template <typename T> class Matrix;

// Class which mimics Math::Matrix class, but holds a reference to a data vector
// instead of the data itself. Hence, it has no ownership of the data it holds.
template <typename T> class MatrixView : public MatrixBase<T> {
public:
  MatrixView() = delete;

  // Get specific item from matrix
  const T &operator[](const size_t row, const size_t col) const;

  // Reshapes matrix view to given dimensions. Returns *this.
  // Throws if given (rows * cols) is not equal to current (rows * cols).
  MatrixView &reshape(const size_t rows, const size_t cols);

  // Getters
  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }

  friend Matrix<T>;

private:
  MatrixView(size_t start, size_t rows, size_t cols, const std::vector<T> &data)
      : m_data{data}, m_start{start}, m_rows{rows}, m_cols{cols} {}

  const std::vector<T> &m_data{};
  size_t m_start{};
  size_t m_rows{};
  size_t m_cols{};
};
} // namespace Math

#include "matrixView.tpp"
