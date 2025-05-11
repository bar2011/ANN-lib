#pragma once

#include "matrix.h"
#include "vector.h"

namespace Math {

// Variations of dot product - vector w/ vector, matrix w/ vector, matrix w/
// matrix
template <typename T>
T dot(const VectorBase<T> &va, const VectorBase<T> &vb,
      bool parallelize = false);
template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const VectorBase<T> &v,
              bool parallelize = false);
template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
              bool parallelize = false);

// Matrix with matrix dot product where the second one is transposed by default
// (needed for ANN calculations and more efficiently done as one function)
template <typename T, typename U, typename V>
Matrix<T> dotTranspose(const MatrixBase<U> &ma, const MatrixBase<V> &mb,
                       bool transposeFirst = false);

// Vector element-wise addition
template <typename T>
VectorBase<T> operator+(const VectorBase<T> &a, const VectorBase<T> &b);

// Column or row vector addition (depends on the given sizes)
template <typename T>
Matrix<T> operator+(const MatrixBase<T> &m, const VectorBase<T> &v);
}; // namespace Math

// Include template function implementation file
#include "linear.tpp"
