// DO NOT INCLUDE THIS FILE - include matrixView.h instead
#pragma once

#include "exception.h"
#include "matrixView.h"

namespace Math {

template <typename T>
const T &MatrixView<T>::operator[](const size_t row, const size_t col) const {
  if (row >= m_rows)
    throw Math::Exception{
        "Math::MatrixView<T>::operator[](const size_t, const size_t) const",
        "Invalid row number: out of bounds"};
  if (col >= m_cols)
    throw Math::Exception{
        "Math::MatrixView<T>::operator[](const size_t, const size_t) const",
        "Invalid column number: out of bounds"};

  return m_data[m_start + row * m_cols + col];
}

template <typename T>
MatrixView<T> &MatrixView<T>::reshape(const size_t rows, const size_t cols) {
  if (rows * cols != m_rows * m_cols)
    throw Math::Exception{
        "Math::MatrixView<T>::reshape(const size_t, const size_t)",
        "Can't reshape a matrix where the given dimensions don't give the same "
        "number of values as the previous dimensions"};

  m_rows = rows;
  m_cols = cols;

  return *this;
}
} // namespace Math
