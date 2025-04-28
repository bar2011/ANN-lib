#pragma once

#include "linear.h"

#include "exception.h"
#include "matrix.h"
#include "vector.h"

namespace Math {

template <typename T> T dot(const Vector<T> &va, const Vector<T> &vb) {
  if (va.size() != vb.size())
    throw Math::Exception{
        "Math::dot(const Vector<T>&, const Vector<T>&)",
        "Can't calculate the dot product of two differently sized vectors"};
  T result{};

  for (size_t i{}; i < va.size(); ++i)
    result += va[i] * vb[i];

  return result;
}

template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const Vector<T> &v) {
  if (m.cols() != v.size())
    throw Math::Exception{
        "Math::dot(const MatrixBase<T>&, const Vector<T>&)",
        "Can't calculate the dot product of a matrix and vector where the "
        "matrix's columns isn't as the vector's size"};

  Vector<T> result{m.rows()};

  for (size_t i{}; i < m.rows(); ++i) {
    T sum{};
    for (size_t j{}; j < m.cols(); ++j)
      sum += m[i, j] * v[j];
    result[i] = sum;
  }

  return result;
}

template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb) {
  if (ma.cols() != mb.rows())
    throw Math::Exception{
        "Math::dot(const MatrixBase<T>&, const MatrixBase<T>&)",
        "Can't compute the dot product of two matrices where the first "
        "matrix's col number isn't the same as the second matrix's row number"};

  Matrix<T> result{ma.rows(), mb.cols()};

  for (size_t i{}; i < ma.rows(); ++i) {
    for (size_t j{}; j < mb.cols(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.cols(); ++k)
        sum += ma[i, k] * mb[k, j];
      result[i, j] = sum;
    }
  }

  return result;
}

template <typename T, typename U, typename V>
Matrix<T> dotTranspose(const MatrixBase<U> &ma, const MatrixBase<V> &mb) {
  if (ma.cols() != mb.cols())
    throw Math::Exception{
        "Math::dotTranspose(const MatrixBase<T>&, const MatrixBase<T>&)",
        "Can't compute the \"transposed\" dot product of two matrices where "
        "the first matrix's col number isn't the same as the second matrix's "
        "col number (as it's rotated)"};

  Matrix<T> result{ma.rows(), mb.rows()};

  for (size_t i{}; i < ma.rows(); ++i) {
    for (size_t j{}; j < mb.rows(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.cols(); ++k)
        sum += ma[i, k] * mb[j, j];
      result[i, j] = sum;
    }
  }

  return static_cast<Matrix<T>>(result);
}

template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b) {
  if (a.size() != b.size())
    throw Math::Exception{"Math::operator+(Vector<T>, const Vector<T>&)",
                          "Unable to add two vectors of different sizes"};

  Vector<T> result(a.size());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

  return result;
}

template <typename T>
Matrix<T> operator+(const MatrixBase<T> &m, const Vector<T> &v) {
  if (m.cols() == v.size()) { // row wise addition
    Matrix<T> result{m.rows(), m.cols()};
    for (size_t i{}; i < m.rows(); ++i)
      for (size_t j{}; j < m.cols(); ++j)
        result[i, j] = m[i, j] + v[j];

    return result;
  }

  if (m.rows() == v.size()) { // column wise addition
    Matrix<T> result{m.rows(), m.cols()};
    for (size_t i{}; i < m.rows(); ++i)
      for (size_t j{}; j < m.cols(); ++j)
        result[i, j] = m[i, j] + v[i];

    return result;
  }

  throw Math::Exception{
      "Math::operator+(const MatrixBase<T>&, const Vector<T>&)",
      "Can't add matrix and vector where their sizes don't match for row or "
      "column wise addition"};
}

}; // namespace Math
