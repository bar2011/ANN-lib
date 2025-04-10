#pragma once

#include "matrix.h"
#include "vector.h"

namespace Math {

// Variations of dot product - vector w/ vector, matrix w/ vector, matrix w/
// matrix
template <typename T> T dot(const Vector<T> &va, const Vector<T> &vb);
template <typename T> Vector<T> dot(const Matrix<T> &m, const Vector<T> &v);
template <typename T> Matrix<T> dot(const Matrix<T> &ma, const Matrix<T> &mb);
// Matrix with matrix dot product where the second one is transposed
// (needed for ANN calculations and more efficiently done as one function)
template <typename T>
Matrix<T> dotTranspose(const Matrix<T> &ma, const Matrix<T> &mb);
}; // namespace Math

// Include template function implementation file
#include "linear.tpp"
