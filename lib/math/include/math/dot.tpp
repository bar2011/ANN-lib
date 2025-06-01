#pragma once

#include "dot.h"

#include "exception.h"
#include "matrix.h"
#include "utils/exceptions.h"
#include "vector.h"

#include "utils/parallel.h"

#include <vector>

namespace Math {

template <typename T>
T dot(const VectorBase<T> &va, const VectorBase<T> &vb,
      std::optional<bool> parallelize) {
  if (va.size() != vb.size())
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Can't calculate the dot product of two differently sized vectors"};
  T result{};

  const size_t size{va.size()};

  // Operation cost per iteration (single multiplication, and vector summation)
  constexpr size_t cost{2};

  // Optimize if explicitly told so or if cost exceeded parallel threshold
  const bool optimize{
      parallelize.value_or(size * cost > PARALLEL_COST_MINIMUM)};
  if (!optimize) {
    for (size_t i{}; i < size; ++i)
      result += va[i] * vb[i];
    return result;
  }

  std::vector<T> partialResults(size);

  Utils::Parallel::parallelFor(size, [&va, &vb, &partialResults](size_t i) {
    partialResults[i] = va[i] * vb[i];
  });

  for (auto &partialResult : partialResults)
    result += partialResult;

  return result;
}

template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const VectorBase<T> &v,
              std::optional<bool> parallelize) {
  if (m.cols() != v.size())
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Can't calculate the dot product of a matrix and vector where the "
        "matrix's columns isn't as the vector's size"};

  Vector<T> result{m.rows()};

  const auto computeRow{[&result, &m, &v](size_t i) {
    T sum{};
    for (size_t j{}; j < m.cols(); ++j)
      sum += m[i, j] * v[j];
    result[i] = sum;
  }};

  // Operation cost per iteration (n additions and multiplications)
  const size_t cost{2 * m.cols()};

  Utils::Parallel::dynamicParallelFor(cost, m.rows(), computeRow, parallelize);

  return result;
}

template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
              std::optional<bool> parallelize,
              std::optional<bool> optimizeCache) {
  if (ma.cols() != mb.rows())
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Can't compute the dot product of two matrices where the first "
        "matrix's col number isn't the same as the second matrix's row number"};

  // Operation cost per iteration (n additions and multiplications)
  const size_t cost{2 * mb.cols() * ma.cols()};

  if (optimizeCache.value_or(ma.rows() * cost > PARALLEL_COST_MINIMUM))
    return dotTB(ma, mb.transpose(), parallelize);

  Matrix<T> result{ma.rows(), mb.cols()};

  const auto computeRow{[&result, &ma, &mb](size_t i) {
    for (size_t j{}; j < mb.cols(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.cols(); ++k)
        sum += ma[i, k] * mb[k, j];
      result[i, j] = sum;
    }
  }};

  Utils::Parallel::dynamicParallelFor(cost, ma.rows(), computeRow, parallelize);

  return result;
}

template <typename T>
Matrix<T> dotTA(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                std::optional<bool> parallelize,
                std::optional<bool> optimizeCache) {
  if (ma.rows() != mb.rows())
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Can't compute the \"transposed\" dot product of two matrices where "
        "the first matrix's row number isn't the same as the second matrix's "
        "row number"};

  // Operation cost per iteration (n additions and multiplications)
  const size_t cost{2 * mb.cols() * ma.rows()};

  if (optimizeCache.value_or(ma.cols() * cost > PARALLEL_COST_MINIMUM))
    return dotTB(ma.transpose(4, parallelize), mb.transpose(4, parallelize),
                 parallelize);

  Matrix<T> result{ma.cols(), mb.cols()};

  const auto computeRow{[&result, &ma, &mb](size_t i) {
    for (size_t j{}; j < mb.cols(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.rows(); ++k)
        sum += ma[k, i] * mb[k, j];
      result[i, j] = sum;
    }
  }};

  Utils::Parallel::dynamicParallelFor(cost, ma.cols(), computeRow, parallelize);

  return result;
}

template <typename T>
Matrix<T> dotTB(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                std::optional<bool> parallelize) {
  if (ma.cols() != mb.cols())
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Can't compute the \"transposed\" dot product of two matrices where "
        "the first matrix's col number isn't the same as the second matrix's "
        "col number"};

  Matrix<T> result{ma.rows(), mb.rows()};

  const auto computeRow{[&result, &ma, &mb](size_t i) {
    for (size_t j = 0; j < mb.rows(); ++j) {
      T sum{};
      size_t k{};

      // Unrolling the loop for better performance
      for (; k + 4 <= ma.cols(); k += 4) {
        sum += ma[i, k] * mb[j, k];
        sum += ma[i, k + 1] * mb[j, k + 1];
        sum += ma[i, k + 2] * mb[j, k + 2];
        sum += ma[i, k + 3] * mb[j, k + 3];
      }
      // Handle remaining elements
      for (; k < ma.cols(); ++k) {
        sum += ma[i, k] * mb[j, k];
      }
      result[i, j] = sum;
    }
  }};

  // Operation cost per iteration (n additions and multiplications)
  const size_t cost{2 * mb.rows() * ma.cols()};

  Utils::Parallel::dynamicParallelFor(cost, ma.rows(), computeRow, parallelize);

  return result;
}

}; // namespace Math
