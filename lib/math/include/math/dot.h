#pragma once

#include "matrix.h"
#include "vector.h"

#include <optional>

namespace Math {

// va - first vector
// vb - second vector
// parallelize - should dot product be parallized. If provided empty, will
//               parallelize automatically as seen needed
template <typename T>
T dot(const VectorBase<T> &va, const VectorBase<T> &vb,
      std::optional<bool> parallelize = std::nullopt);

// m - matrix
// v - vector
// parallelize - should dot product be parallized. If provided empty, will
//               parallelize automatically as seen needed
template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const VectorBase<T> &v,
              std::optional<bool> parallelize = std::nullopt);

// ma - first matrix
// mb - second matrix
// parallelize - should dot product be parallized. If provided empty, will
//               parallelize automatically as seen needed
// optimizeCache - should matrices be transposed as needed to optimize cache
//                 friendliness. If provided empty, will transpose as seen
//                 needed. the only real reason to turn this off is if memory is
//                 a real concern
template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
              std::optional<bool> parallelize = std::nullopt,
              std::optional<bool> optimizeCache = std::nullopt);

// dot(a^T, b)
// ma - first matrix
// mb - second matrix
// parallelize - should dot product be parallized. If provided empty, will
//               parallelize automatically as seen needed
// optimizeCache - should matrices be transposed as needed to optimize cache
//                 friendliness. If provided empty, will transpose as seen
//                 needed. the only real reason to turn this off is if memory is
//                 a real concern
template <typename T>
Matrix<T> dotTA(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                std::optional<bool> parallelize = std::nullopt,
                std::optional<bool> optimizeCache = std::nullopt);

// dot(a, b^T)
// va - first vector
// vb - second vector
// parallelize - should dot product be parallized. If provided empty, will
//               parallelize automatically as seen needed
template <typename T>
Matrix<T> dotTB(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                std::optional<bool> parallelize = std::nullopt);

}; // namespace Math

// Include template function implementation file
#include "dot.tpp"
