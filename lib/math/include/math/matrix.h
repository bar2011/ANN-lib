#pragma once

#include "matrixBase.h"
#include "matrixView.h"
#include "vector.h"

#include <functional>
#include <memory>
#include <vector>

namespace Math {

// Simple matrix class with access to mathematical operations
// T = the data type the matrix holds
template <typename T> class Matrix : public MatrixBase<T> {
public:
  Matrix() : m_rows{0}, m_cols{0}, m_data(0) {};

  // Create 0-filled matrix of given size
  Matrix(const size_t rows, const size_t cols);

  // Create matrix filled with values outputted from given function
  Matrix(const size_t rows, const size_t cols, std::function<T()> gen);

  // Constructor with rows, columns, and a single  data array of type T of size
  // (rows * cols) containing all the matrix's data
  Matrix(const size_t rows, const size_t cols, const T *const data);

  // Copy constructor
  Matrix(const Matrix &other);

  // Move constructor
  Matrix(Matrix &&other) noexcept;

  // Copy assignment
  Matrix &operator=(const Matrix &other);

  // Move assignment
  Matrix &operator=(Matrix &&other) noexcept;

  // Fill the matrix with values from the generator function
  // gen input - a pointer to the item to be filled
  void fill(std::function<void(T *)> gen);

  // Get specific item from matrix
  T &operator[](const size_t row, const size_t col);
  const T &operator[](const size_t row, const size_t col) const;

  // Reshapes matrix to given dimensions. Returns *this.
  // Throws if given (rows * cols) is not equal to current (rows * cols).
  Matrix &reshape(const size_t rows, const size_t cols);

  // Get view of the entire matrix.
  std::shared_ptr<MatrixView<T>> view() const;

  // Returns a view of a range of rows from the matrix.
  // Includes rows in the range [startRow, endRow), i.e., startRow is inclusive,
  // endRow is exclusive. The view includes all columns in each row.
  // Throws if endRow > row count or startRow >= endRow.
  std::shared_ptr<MatrixView<T>> view(size_t startRow, size_t endRow) const;

  // Returns the index of the biggest value in the matrix
  // format: (row, col)
  std::pair<size_t, size_t> argmax() const;

  // Returns a vector containing the index of the biggest value in each row
  std::unique_ptr<Math::Vector<size_t>> argmaxRow() const;

  // Returns a vector containing the index of the biggest value in each column
  std::unique_ptr<Math::Vector<size_t>> argmaxCol() const;

  // Getters
  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }

private:
  std::vector<T> m_data{};
  size_t m_rows{};
  size_t m_cols{};
};
}; // namespace Math

#include "matrix.tpp"
