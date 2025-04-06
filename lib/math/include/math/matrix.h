#pragma once

#include "vector.h"
#include <functional>
#include <vector>

namespace Math {

// Simple matrix class with access to mathematical operations
// T = the data type the matrix holds
template <typename T> class Matrix {
public:
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

  // Get specific item from matrix
  T &operator[](const size_t row, const size_t col);
  const T &operator[](const size_t row, const size_t col) const;

  Vector<T> *flatten();

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
