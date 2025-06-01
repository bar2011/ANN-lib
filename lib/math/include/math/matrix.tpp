#pragma once

#include "exception.h"
#include "matrix.h"
#include "utils/exceptions.h"
#include "vector.h"
#include "vectorView.h"

#include "utils/parallel.h"

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
Matrix<T>::Matrix(const size_t rows, const size_t cols, std::vector<T> &&data)
    : m_data{std::move(data)}, m_rows{rows}, m_cols{cols} {
  if (m_data.size() != m_rows * m_cols)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Given vector doesn't match rows * cols"};
}

template <typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols)
    : m_data(rows * cols), m_rows{rows}, m_cols{cols} {};

template <typename T>
Matrix<T>::Matrix(const MatrixBase<T> &other)
    : m_data(other.data().size()), m_rows{other.rows()}, m_cols{other.cols()} {
  std::copy_n(other.data().begin(), m_rows * m_cols, m_data.begin());
}

template <typename T>
Matrix<T>::Matrix(Matrix &&other) noexcept
    : m_data{std::move(other.m_data)}, m_rows{other.m_rows},
      m_cols{other.m_cols} {
  other.m_rows = 0;
  other.m_cols = 0;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix &other) {
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

template <typename T>
void Matrix<T>::fill(std::function<void(T *)> gen,
                     std::optional<bool> parallelize, size_t cost) {
  Utils::Parallel::dynamicParallelFor(
      cost, m_data.size(), [gen, &data = m_data](size_t i) { gen(&data[i]); },
      parallelize);
}

template <typename T>
void Matrix<T>::transform(const MatrixBase<T> &m,
                          std::function<void(T *, const T *)> gen,
                          std::optional<bool> parallelize, size_t cost) {
  if (m.rows() != rows() || m.cols() != cols())
    throw Math::Exception{CURRENT_FUNCTION,
                          "Can't transform matrices with different dimensions"};
  Utils::Parallel::dynamicParallelFor(
      cost, m_data.size(),
      [&gen, &data = m_data, &m](size_t i) { gen(&data[i], &m.data()[i]); });
}

template <typename T>
void Matrix<T>::transform(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                          std::function<void(T *, const T *, const T *)> gen,
                          std::optional<bool> parallelize, size_t cost) {
  if (ma.rows() != rows() || ma.cols() != cols() || mb.rows() != rows() ||
      mb.cols() != cols())
    throw Math::Exception{CURRENT_FUNCTION,
                          "Can't transform matrices with different dimensions"};
  Utils::Parallel::dynamicParallelFor(
      cost, m_data.size(), [&gen, &data = m_data, &ma, &mb](size_t i) {
        gen(&data[i], &ma.data()[i], &mb.data()[i]);
      });
}

template <typename T> void Matrix<T>::insertRow(const Vector<T> &v) {
  if (v.size() != cols())
    throw Math::Exception{CURRENT_FUNCTION, "Invalid size for added row"};

  ++m_rows;
  m_data.insert(m_data.end(), v.m_data.begin(), v.m_data.end());
}

template <typename T>
std::shared_ptr<Matrix<T>>
Matrix<T>::transpose(size_t chunkSize, std::optional<bool> parallelize) const {
  auto result{std::make_shared<Matrix<T>>(cols(), rows())};

  auto srcData{m_data.begin()};
  auto resultData{result->m_data.begin()};
  const size_t cost{cols() * chunkSize * chunkSize};

  Utils::Parallel::dynamicParallelFor(
      cost, (rows() + chunkSize - 1) / chunkSize,
      [srcData, resultData, cols = m_cols, rows = m_rows, chunkSize](size_t i) {
        size_t ciStart{i * chunkSize};
        size_t ciEnd{std::min(ciStart + chunkSize, rows)};

        for (size_t j{}; j < (cols + chunkSize - 1) / chunkSize; ++j) {
          size_t cjStart{j * chunkSize};
          size_t cjEnd{std::min(cjStart + chunkSize, cols)};

          // Process a single block of chunkSize x chunkSize
          for (size_t ci{ciStart}; ci < ciEnd; ++ci)
            for (size_t cj{cjStart}; cj < cjEnd; ++cj)
              resultData[cj * rows + ci] = srcData[ci * cols + cj];
        }
      },
      parallelize);

  return result;
}

template <typename T>
T &Matrix<T>::operator[](const size_t row, const size_t col) {
  return m_data[row * m_cols + col];
}

template <typename T>
const T &Matrix<T>::operator[](const size_t row, const size_t col) const {
  return m_data[row * m_cols + col];
}

template <typename T> T &Matrix<T>::at(const size_t row, const size_t col) {
  if (row >= m_rows)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Invalid row number: out of bounds"};
  if (col >= m_cols)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Invalid column number: out of bounds"};

  return m_data[row * m_cols + col];
}

template <typename T>
const T &Matrix<T>::at(const size_t row, const size_t col) const {
  if (row >= m_rows)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Invalid row number: out of bounds"};
  if (col >= m_cols)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Invalid column number: out of bounds"};

  return m_data[row * m_cols + col];
}

template <typename T>
const std::shared_ptr<VectorView<T>> Matrix<T>::operator[](const size_t row) {
  return std::make_shared<VectorView<T>>(row * m_cols, m_rows, m_data);
}

template <typename T>
const std::shared_ptr<const VectorView<T>>
Matrix<T>::operator[](const size_t row) const {
  return std::make_shared<VectorView<T>>(row * m_cols, m_rows, m_data);
}

template <typename T>
const std::shared_ptr<VectorView<T>> Matrix<T>::at(const size_t row) {
  if (row >= m_rows)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Invalid row number: out of bounds"};

  return std::make_shared<VectorView<T>>(row * m_cols, m_rows, m_data);
}

template <typename T>
const std::shared_ptr<const VectorView<T>>
Matrix<T>::at(const size_t row) const {
  if (row >= m_rows)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Invalid row number: out of bounds"};

  return std::make_shared<VectorView<T>>(row * m_cols, m_rows, m_data);
}

template <typename T>
Matrix<T> &Matrix<T>::reshape(const size_t rows, const size_t cols) {
  if (rows * cols != m_rows * m_cols)
    throw Math::Exception{CURRENT_FUNCTION, "Reshape dimension mismatch"};

  m_rows = rows;
  m_cols = cols;

  return *this;
};

template <typename T> std::shared_ptr<MatrixView<T>> Matrix<T>::view() const {
  return std::shared_ptr<MatrixView<T>>(
      new MatrixView<T>{0, m_rows, m_cols, m_data});
}

template <typename T>
std::shared_ptr<MatrixView<T>> Matrix<T>::view(size_t startRow,
                                               size_t endRow) const {
  if (startRow >= endRow)
    throw Math::Exception{CURRENT_FUNCTION, "Start row ahead of the end row"};
  if (endRow > m_rows)
    throw Math::Exception{CURRENT_FUNCTION,
                          "End row is outside the matrix's bound"};

  // Can't use make_shared because the constructor is private
  return std::shared_ptr<MatrixView<T>>{
      new MatrixView<T>{startRow * m_cols, endRow - startRow, m_cols, m_data}};
}

template <typename T> std::pair<size_t, size_t> Matrix<T>::argmax() const {
  if (rows() == 0 || cols() == 0)
    throw Math::Exception{CURRENT_FUNCTION,
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
std::unique_ptr<Math::Vector<size_t>> Matrix<T>::argmaxRow() const {
  if (rows() == 0 || cols() == 0)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Can't get the maximum of an empty matrix"};

  auto maxRow{std::make_unique<Math::Vector<size_t>>(rows())};

  for (size_t i{}; i < rows(); ++i)
    for (size_t j{}; j < cols(); ++j)
      if (operator[](i, (*maxRow)[i]) < operator[](i, j))
        (*maxRow)[i] = j;

  return maxRow;
}

template <typename T>
std::unique_ptr<Math::Vector<size_t>> Matrix<T>::argmaxCol() const {
  if (rows() == 0 || cols() == 0)
    throw Math::Exception{CURRENT_FUNCTION,
                          "Can't get the maximum of an empty matrix"};

  auto maxCol{std::make_unique<Math::Vector<size_t>>(cols())};

  for (size_t i{}; i < rows(); ++i)
    for (size_t j{}; j < cols(); ++j)
      if (operator[]((*maxCol)[j], j) < operator[](i, j))
        (*maxCol)[j] = i;

  return maxCol;
}

template <typename T>
std::unique_ptr<Math::VectorView<T>> Matrix<T>::asVector() {
  return std::unique_ptr<Math::VectorView<T>>{
      new Math::VectorView<T>{0, m_rows * m_cols, m_data}};
}
}; // namespace Math
