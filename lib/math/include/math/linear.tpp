#pragma once

#include "linear.h"

#include "exception.h"
#include "matrix.h"
#include "vector.h"

#include <algorithm>

namespace Math {

template <typename T>
Vector<T> operator+(const VectorBase<T> &a, const VectorBase<T> &b) {
  if (a.size() != b.size())
    throw Math::Exception{
        "Math::operator+(VectorBase<T>, const VectorBase<T>&)",
        "Unable to add two vectors of different sizes"};

  Vector<T> result(a.size());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

  return result;
}

template <typename T>
Matrix<T> operator+(const MatrixBase<T> &m, const VectorBase<T> &v) {
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
      "Math::operator+(const MatrixBase<T>&, const VectorBase<T>&)",
      "Can't add matrix and vector where their sizes don't match for row or "
      "column wise addition"};
}
}; // namespace Math
