#pragma once

#include "exception.h"
#include "matrixView.h"

#include "utils/parallel.h"

namespace Math {

template <typename T>
const T &MatrixView<T>::operator[](const size_t row, const size_t col) const {
  return m_data[m_start + row * m_cols + col];
}

template <typename T>
const T &MatrixView<T>::at(const size_t row, const size_t col) const {
  if (row >= m_rows)
    throw Math::Exception{
        "Math::MatrixView<T>::at(const size_t, const size_t) const",
        "Invalid row number: out of bounds"};
  if (col >= m_cols)
    throw Math::Exception{
        "Math::MatrixView<T>::at(const size_t, const size_t) const",
        "Invalid column number: out of bounds"};

  return m_data[m_start + row * m_cols + col];
}

template <typename T>
MatrixView<T> &MatrixView<T>::reshape(const size_t rows, const size_t cols) {
  if (rows * cols != m_rows * m_cols)
    throw Math::Exception{
        "Math::MatrixView<T>::reshape(const size_t, const size_t)",
        "Reshape dimension mismatch"};

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

template <typename T>
std::shared_ptr<MatrixView<T>> MatrixView<T>::view(size_t startRow,
                                                   size_t endRow) const {
  if (startRow >= endRow)
    throw Math::Exception{"Math::MatrixView<T>::view(size_t, size_t) const",
                          "Start row ahead of the end row"};
  if (endRow > m_rows)
    throw Math::Exception{"Math::MatrixView<T>::view(size_t, size_t) const",
                          "End row is outside the matrix's bound"};

  // Can't use make_shared because the constructor is private
  return std::shared_ptr<MatrixView<T>>{
      new MatrixView<T>{m_start + startRow * m_cols,
                        m_start + endRow - startRow, m_cols, m_data}};
}

template <typename T>
std::shared_ptr<Matrix<T>>
MatrixView<T>::transpose(size_t chunkSize,
                         std::optional<bool> parallelize) const {
  auto result{std::make_shared<Matrix<T>>(cols(), rows())};
  const size_t cost{cols() * chunkSize * chunkSize};

  Utils::Parallel::dynamicParallelFor(
      cost, (rows() + chunkSize - 1) / chunkSize,
      [this, &result, chunkSize](size_t i) {
        for (size_t j{}; j < cols() / chunkSize; ++j)
          // Process a single block of chunkSizexchunkSize
          for (size_t ci{i * chunkSize};
               ci < std::min((i + 1) * chunkSize, rows()); ++ci)
            for (size_t cj{j * chunkSize};
                 cj < std::min((j + 1) * chunkSize, cols()); ++cj)
              (*result)[cj, ci] = operator[](ci, cj);
      },
      parallelize);

  return result;
}
} // namespace Math
