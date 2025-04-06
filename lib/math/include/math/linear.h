#pragma once

#include "matrix.h"
#include "vector.h"

namespace Math {

template <typename T> T dot(const Vector<T> &va, const Vector<T> &vb);
template <typename T> Vector<T> dot(const Matrix<T> &m, const Vector<T> &v);
}; // namespace Math

// Include template function implementation file
#include "linear.tpp"
