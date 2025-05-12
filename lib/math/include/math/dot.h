#pragma once

#include "matrix.h"
#include "vector.h"

namespace Math {

template <typename T>
T dot(const VectorBase<T> &va, const VectorBase<T> &vb,
      bool parallelize = false);

template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const VectorBase<T> &v,
              bool parallelize = false);

template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
              bool parallelize = false);

// dot(a^T, b)
template <typename T, typename U, typename V>
Matrix<T> dotTA(const MatrixBase<U> &ma, const MatrixBase<V> &mb,
                bool parallelize = false);

// dot(a, b^T)
template <typename T, typename U, typename V>
Matrix<T> dotTB(const MatrixBase<U> &ma, const MatrixBase<V> &mb,
                bool parallelize = false);

}; // namespace Math

// Include template function implementation file
#include "dot.tpp"
