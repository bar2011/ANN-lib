#pragma once

#include "matrix.h"
#include "vectorBase.h"

namespace Math {

// Vector element-wise addition
template <typename T>
VectorBase<T> operator+(const VectorBase<T> &a, const VectorBase<T> &b);

// Column or row vector addition (depends on the given sizes)
template <typename T>
Matrix<T> operator+(const MatrixBase<T> &m, const VectorBase<T> &v);

}; // namespace Math

// Include template function implementation file
#include "linear.tpp"
