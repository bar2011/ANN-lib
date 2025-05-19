#pragma once

#include "matrixBase.h"
#include "vector.h"

#include <vector>

namespace Math {

template <typename T> class Matrix;

// Class which mimics Math::Matrix class, but holds a reference to a data vector
// instead of the data itself. Hence, it has no ownership of the data it holds.
template <typename T> class MatrixView : public MatrixBase<T> {
public:
  MatrixView() = delete;

  // Single item access - NO BOUNDS CHECKING
  const T &operator[](const size_t row, const size_t col) const;
  // Single item access - WITH BOUNDS CHECKING
  const T &at(const size_t row, const size_t col) const;

  // Reshapes matrix view to given dimensions. Returns *this.
  // Throws if given (rows * cols) is not equal to current (rows * cols).
  MatrixView &reshape(const size_t rows, const size_t cols);

  // Getters
  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }

  // Return entire underlying data. Not necessarily from the start of MatrixView
  const std::vector<T> &data() const { return m_data; };

  // Returns the index of the biggest value in the matrix
  // format: (row, col)
  std::pair<size_t, size_t> argmax() const;

  // Returns a vector containing the index of the biggest value in each row
  std::unique_ptr<Math::Vector<size_t>> argmaxRow() const;

  // Returns a vector containing the index of the biggest value in each column
  std::unique_ptr<Math::Vector<size_t>> argmaxCol() const;

  // Transposes the matrix. Returns the transposed one.
  // Note: the returned matrix has complete ownership on its values
  std::shared_ptr<Matrix<T>>
  transpose(size_t chunkSize = 4,
            std::optional<bool> parallelize = std::nullopt) const;

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
