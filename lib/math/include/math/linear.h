#pragma once

#include "matrix.h"
#include "matrixBase.h"
#include "vector.h"

namespace Math {

// Variations of dot product - vector w/ vector, matrix w/ vector, matrix w/
// matrix
template <typename T> T dot(const Vector<T> &va, const Vector<T> &vb);
template <typename T> Vector<T> dot(const MatrixBase<T> &m, const Vector<T> &v);
template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb);

// Matrix with matrix dot product where the second one is transposed
// (needed for ANN calculations and more efficiently done as one function)
template <typename T, typename U, typename V>
Matrix<T> dotTranspose(const MatrixBase<U> &ma, const MatrixBase<V> &mb);

// Vector element-wise addition
template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b);

// Column or row vector addition (depends on the given sizes)
template <typename T>
Matrix<T> operator+(const MatrixBase<T> &m, const Vector<T> &v);
}; // namespace Math

// Include template function implementation file
#include "linear.tpp"
