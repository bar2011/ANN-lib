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

template <typename T> std::pair<size_t, size_t> MatrixView<T>::argmax() const {
  if (rows() == 0 || cols() == 0)
    throw Math::Exception{"Math::MatrixView<T>::argmax() const",
                          "Can't get the maximum of an empty matrix"};

  auto maxIndex{std::pair{0uz, 0uz}};

  for (size_t i{}; i < rows(); ++i)
    for (size_t j{}; j < cols(); ++j)
      if (operator[](std::get<0>(maxIndex),
                     std::get<1>(maxIndex)) < operator[](i, j))
        maxIndex = {i, j};

  return maxIndex;
}

template <typename T>
std::unique_ptr<Math::Vector<size_t>> MatrixView<T>::argmaxRow() const {
  if (rows() == 0 || cols() == 0)
    throw Math::Exception{"Math::MatrixView<T>::argmaxRow() const",
                          "Can't get the maximum of an empty matrix"};

  auto maxRow{std::make_unique<Math::Vector<size_t>>(rows())};

  for (size_t i{}; i < rows(); ++i)
    for (size_t j{}; j < cols(); ++j)
      if (operator[](i, (*maxRow)[i]) < operator[](i, j))
        (*maxRow)[i] = j;

  return maxRow;
}

template <typename T>
std::unique_ptr<Math::Vector<size_t>> MatrixView<T>::argmaxCol() const {
  if (rows() == 0 || cols() == 0)
    throw Math::Exception{"Math::MatrixView<T>::argmaxCol() const",
                          "Can't get the maximum of an empty matrix"};

  auto maxCol{std::make_unique<Math::Vector<size_t>>(cols())};

  for (size_t i{}; i < rows(); ++i)
    for (size_t j{}; j < cols(); ++j)
      if (operator[]((*maxCol)[j], j) < operator[](i, j))
        (*maxCol)[j] = i;

  return maxCol;
}
} // namespace Math
