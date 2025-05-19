#pragma once

#include "linear.h"

#include "exception.h"
#include "matrix.h"
#include "vector.h"

#include "utils/parallel.h"

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
  Matrix<T> result{m.rows(), m.cols()};

  // m.cols() different additions
  const size_t cost{m.cols()};

  std::function<void(size_t)> computeRow;

  if (m.cols() == v.size()) // row wise addition
    computeRow = [&result, &m, &v](size_t i) {
      for (size_t j{}; j < m.cols(); ++j)
        result[i, j] = m[i, j] + v[j];
    };
  else if (m.rows() == v.size()) // column wise addition
    computeRow = [&result, &m, &v](size_t i) {
      for (size_t j{}; j < m.cols(); ++j)
        result[i, j] = m[i, j] + v[i];
    };
  else
    throw Math::Exception{
        "Math::operator+(const MatrixBase<T>&, const VectorBase<T>&)",
        "Can't add matrix and vector where their sizes don't match for row or "
        "column wise addition"};

  Utils::Parallel::dynamicParallelFor(cost, m.rows(), computeRow);

  return result;
}
}; // namespace Math
