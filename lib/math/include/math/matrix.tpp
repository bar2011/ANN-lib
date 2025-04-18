#pragma once

#include "exception.h"
#include "matrix.h"

#include <algorithm>
#include <utility>

namespace Math {

template <typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, std::function<T()> gen)
    : m_data(rows * cols), m_rows{rows}, m_cols{cols} {
  std::generate_n(m_data.begin(), rows * cols, gen);
}

template <typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, const T *const data)
    : m_data(rows * cols), m_rows{rows}, m_cols{cols} {
  std::copy_n(data, rows * cols, m_data.begin());
}

template <typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols)
    : m_data(rows * cols), m_rows{rows}, m_cols{cols} {};

template <typename T>
Matrix<T>::Matrix(const Matrix &other)
    : m_data(other.m_data.size()), m_rows{other.m_rows}, m_cols{other.m_cols} {
  std::copy_n(other.m_data.begin(), m_rows * m_cols, m_data.begin());
}

template <typename T>
Matrix<T>::Matrix(Matrix &&other) noexcept
    : m_data{std::move(other.m_data)}, m_rows{other.m_rows},
      m_cols{other.m_cols} {
  other.m_rows = 0;
  other.m_cols = 0;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix &other) {
  if (other.rows() != this->rows() || other.cols() != this->cols())
    throw Math::Exception{"Matrix<T>::operator=(const Matrix&)",
                          "Matrices dimensions doesn't match."};
  if (&other != this) {
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_data.resize(m_rows * m_cols);
    std::copy_n(other.m_data.begin(), m_rows * m_cols, m_data.begin());
  }
  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(Matrix &&other) noexcept {
  if (&other != this) {
    m_data = std::move(other.m_data);
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    other.m_rows = 0;
    other.m_cols = 0;
  }
  return *this;
}

template <typename T> void Matrix<T>::fill(std::function<void(T *)> gen) {
  for (size_t i{}; i < m_data.size(); ++i)
    gen(&m_data[i]);
}

template <typename T>
T &Matrix<T>::operator[](const size_t row, const size_t col) {
  if (row >= m_rows)
    throw Math::Exception{
        "Math::Matrix<T>::operator[](const size_t, const size_t)",
        "Invalid row number: out of bounds"};
  if (col >= m_cols)
    throw Math::Exception{
        "Math::Matrix<T>::operator[](const size_t, const size_t)",
        "Invalid column number: out of bounds"};

  return m_data[row * m_cols + col];
}

template <typename T>
const T &Matrix<T>::operator[](const size_t row, const size_t col) const {
  if (row >= m_rows)
    throw Math::Exception{
        "Math::Matrix<T>::operator[](const size_t, const size_t) const",
        "Invalid row number: out of bounds"};
  if (col >= m_cols)
    throw Math::Exception{
        "Math::Matrix<T>::operator[](const size_t, const size_t) const",
        "Invalid column number: out of bounds"};

  return m_data[row * m_cols + col];
}

template <typename T>
Matrix<T> &Matrix<T>::reshape(const size_t rows, const size_t cols) {
  if (rows * cols != m_rows * m_cols)
    throw Math::Exception{
        "Math::Matrix<T>::reshape(const size_t, const size_t)",
        "Can't reshape a matrix where the given dimensions don't give the same "
        "number of values as the previous dimensions"};

  m_rows = rows;
  m_cols = cols;

  return *this;
};

template <typename T> MatrixView<T> Matrix<T>::view() const {
  return MatrixView<T>{0, m_rows, m_cols, m_data};
}

template <typename T>
MatrixView<T> Matrix<T>::view(size_t startRow, size_t endRow) const {
  if (startRow >= endRow)
    throw Math::Exception{
        "Math::Matrix<T>::view(size_t, size_t) const",
        "Can't create a row view where the start row is ahead of the end row"};
  if (endRow > m_rows)
    throw Math::Exception{"Math::Matrix<T>::view(size_t, size_t) const",
                          "Can't create a row view where the end row is "
                          "outside the matrix's bound"};
  return MatrixView<T>{startRow * m_cols, endRow - startRow, m_cols, m_data};
}
}; // namespace Math
