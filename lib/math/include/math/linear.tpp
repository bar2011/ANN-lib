// DO NOT INCLUDE THIS FILE - include linear.h instead
#pragma once

#include "linear.h"

#include "exception.h"
#include "matrix.h"
#include "vector.h"

namespace Math {

template <typename T> T dot(const Vector<T> &va, const Vector<T> &vb) {
  if (va.size() != vb.size())
    throw Math::Exception{
        "T dot(const Vector<T>&, const Vector<T>&)",
        "Can't calculate the dot product of two differently sized vectors"};
  T result{};

  for (size_t i{}; i < va.size(); ++i)
    result += va[i] * vb[i];

  return result;
}

template <typename T> Vector<T> dot(const Matrix<T> &m, const Vector<T> &v) {
  if (m.cols() != v.size())
    throw Math::Exception{
        "Vector<T> dot(const Matrix<T>&, const Vector<T>&)",
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
}; // namespace Math
