#pragma once

#include "dot.h"

#include "exception.h"
#include "matrix.h"
#include "vector.h"

#include "utils/parallel.h"

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

namespace Math {

template <typename T>
T dot(const VectorBase<T> &va, const VectorBase<T> &vb, bool parallelize) {
  if (va.size() != vb.size())
    throw Math::Exception{
        "Math::dot(const VectorBase<T>&, const VectorBase<T>&, bool)",
        "Can't calculate the dot product of two differently sized vectors"};
  T result{};

  if (!parallelize) {
    for (size_t i{}; i < va.size(); ++i)
      result += va[i] * vb[i];
    return result;
  }

  std::vector<T> partialResults(va.size());

  Utils::Parallel::parallelFor(va.size(),
                               [&va, &vb, &partialResults](size_t i) {
                                 partialResults[i] += va[i] * vb[i];
                               });

  for (auto &partialResult : partialResults)
    result += partialResult;

  return result;
}

template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const VectorBase<T> &v,
              bool parallelize) {
  if (m.cols() != v.size())
    throw Math::Exception{
        "Math::dot(const MatrixBase<T>&, const VectorBase<T>&, bool)",
        "Can't calculate the dot product of a matrix and vector where the "
        "matrix's columns isn't as the vector's size"};

  Vector<T> result{m.rows()};

  if (!parallelize) {
    for (size_t i{}; i < m.rows(); ++i) {
      T sum{};
      for (size_t j{}; j < m.cols(); ++j)
        sum += m[i, j] * v[j];
      result[i] = sum;
    }

    return result;
  }

  Utils::Parallel::parallelFor(m.rows(), [&m, &v, &result](size_t i) {
    T sum{};
    for (size_t j{}; j < m.cols(); ++j)
      sum += m[i, j] * v[j];
    result[i] = sum;
  });

  return result;
}

template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
              bool parallelize) {
  if (ma.cols() != mb.rows())
    throw Math::Exception{
        "Math::dot(const MatrixBase<T>&, const MatrixBase<T>&, bool)",
        "Can't compute the dot product of two matrices where the first "
        "matrix's col number isn't the same as the second matrix's row number"};

  Matrix<T> result{ma.rows(), mb.cols()};

  if (!parallelize) {
    for (size_t i{}; i < ma.rows(); ++i) {
      for (size_t j{}; j < mb.cols(); ++j) {
        T sum{};
        for (size_t k{}; k < ma.cols(); ++k)
          sum += ma[i, k] * mb[k, j];
        result[i, j] = sum;
      }
    }

    return result;
  }

  Utils::Parallel::parallelFor(ma.rows(), [&ma, &mb, &result](size_t i) {
    for (size_t j{}; j < mb.cols(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.cols(); ++k)
        sum += ma[i, k] * mb[k, j];
      result[i, j] = sum;
    }
  });

  return result;
}

template <typename T>
Matrix<T> dotTA(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                bool parallelize) {
  if (ma.rows() != mb.rows())
    throw Math::Exception{
        "Math::TA(const MatrixBase<T>&, const MatrixBase<T>&, bool)",
        "Can't compute the \"transposed\" dot product of two matrices where "
        "the first matrix's row number isn't the same as the second matrix's "
        "row number"};

  Matrix<T> result{ma.cols(), mb.cols()};

  if (!parallelize) {
    for (size_t i{}; i < ma.cols(); ++i) {
      for (size_t j{}; j < mb.cols(); ++j) {
        T sum{};
        for (size_t k{}; k < ma.rows(); ++k)
          sum += ma[k, i] * mb[k, j];
        result[i, j] = sum;
      }
    }

    return result;
  }

  Utils::Parallel::parallelFor(ma.cols(), [&ma, &mb, &result](size_t i) {
    for (size_t j{}; j < mb.cols(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.rows(); ++k)
        sum += ma[k, i] * mb[k, j];
      result[i, j] = sum;
    }
  });

  return result;
}

template <typename T>
Matrix<T> dotTB(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
                bool parallelize) {
  if (ma.cols() != mb.cols())
    throw Math::Exception{
        "Math::dotTB(const MatrixBase<T>&, const MatrixBase<T>&, bool)",
        "Can't compute the \"transposed\" dot product of two matrices where "
        "the first matrix's col number isn't the same as the second matrix's "
        "col number"};

  Matrix<T> result{ma.rows(), mb.rows()};

  if (!parallelize) {
    for (size_t i{}; i < ma.rows(); ++i) {
      for (size_t j{}; j < mb.rows(); ++j) {
        T sum{};
        for (size_t k{}; k < ma.cols(); ++k)
          sum += ma[i, k] * mb[j, k];
        result[i, j] = sum;
      }
    }

    return result;
  }

  Utils::Parallel::parallelFor(ma.rows(), [&ma, &mb, &result](size_t i) {
    for (size_t j{}; j < mb.rows(); ++j) {
      T sum{};
      for (size_t k{}; k < ma.cols(); ++k)
        sum += ma[i, k] * mb[j, k];
      result[i, j] = sum;
    }
  });

  return result;
}

}; // namespace Math
