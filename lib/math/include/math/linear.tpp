#pragma once

#include "linear.h"

#include "exception.h"
#include "matrix.h"
#include "vector.h"

#include <algorithm>
#include <future>
#include <thread>
#include <vector>

namespace Math {

template <typename T>
T dot(const VectorBase<T> &va, const VectorBase<T> &vb, bool parallelize) {
  if (va.size() != vb.size())
    throw Math::Exception{
        "Math::dot(const VectorBase<T>&, const VectorBase<T>&)",
        "Can't calculate the dot product of two differently sized vectors"};
  T result{};

  if (!parallelize) {
    for (size_t i{}; i < va.size(); ++i)
      result += va[i] * vb[i];
    return result;
  }

  size_t availableThreads{std::clamp<size_t>(
      static_cast<size_t>(std::thread::hardware_concurrency()), 1, va.size())};
  size_t chunkSize{(va.size() + availableThreads - 1) / availableThreads};

  std::vector<std::future<T>> futureResults{};

  for (size_t thread{}; thread < availableThreads; ++thread) {
    size_t start{thread * chunkSize};
    size_t end{std::min((thread + 1) * chunkSize, va.size())};

    futureResults.emplace_back(
        std::async(std::launch::async, [start, end, &va, &vb]() {
          T partialResult{};
          for (size_t i{start}; i < end; ++i)
            partialResult += va[i] * vb[i];
          return partialResult;
        }));
  }

  for (auto &future : futureResults)
    result += future.get();

  return result;
}

template <typename T>
Vector<T> dot(const MatrixBase<T> &m, const VectorBase<T> &v,
              bool parallelize) {
  if (m.cols() != v.size())
    throw Math::Exception{
        "Math::dot(const MatrixBase<T>&, const VectorBase<T>&)",
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

  size_t availableThreads{std::clamp<size_t>(
      static_cast<size_t>(std::thread::hardware_concurrency()), 1, m.rows())};
  size_t chunkSize{(m.rows() + availableThreads - 1) / availableThreads};

  std::vector<std::thread> threads{};

  for (size_t thread{}; thread < availableThreads; ++thread) {
    size_t start{thread * chunkSize};
    size_t end{std::min((thread + 1) * chunkSize, m.rows())};

    threads.emplace_back(std::thread([start, end, thread, &m, &v, &result]() {
      for (size_t i{start}; i < end; ++i) {
        T sum{};
        for (size_t j{}; j < m.cols(); ++j)
          sum += m[i, j] * v[j];
        result[i] = sum;
      }
    }));
  }

  for (auto &t : threads)
    t.join();

  return result;
}

template <typename T, typename U, typename V>
Matrix<T> dot(const MatrixBase<U> &ma, const MatrixBase<V> &mb) {
template <typename T>
Matrix<T> dot(const MatrixBase<T> &ma, const MatrixBase<T> &mb,
              bool parallelize) {
  if (ma.cols() != mb.rows())
    throw Math::Exception{
        "Math::dot(const MatrixBase<T>&, const MatrixBase<T>&)",
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

  size_t availableThreads{std::clamp<size_t>(
      static_cast<size_t>(std::thread::hardware_concurrency()), 1, ma.rows())};
  size_t chunkSize{(ma.rows() + availableThreads - 1) / availableThreads};

  std::vector<std::thread> threads{};

  for (size_t thread{}; thread < availableThreads; ++thread) {
    size_t start{thread * chunkSize};
    size_t end{std::min((thread + 1) * chunkSize, ma.rows())};

    threads.emplace_back(std::thread([start, end, thread, &ma, &mb, &result]() {
      for (size_t i{start}; i < end; ++i) {
        for (size_t j{}; j < mb.cols(); ++j) {
          T sum{};
          for (size_t k{}; k < ma.cols(); ++k)
            sum += ma[i, k] * mb[k, j];
          result[i, j] = sum;
        }
      }
    }));
  }

  for (auto &t : threads)
    t.join();

  return result;
}

template <typename T, typename U, typename V>
Matrix<T> dotTranspose(const MatrixBase<U> &ma, const MatrixBase<V> &mb,
                       bool transposeFirst) {
  if (transposeFirst) {
    if (ma.rows() != mb.rows())
      throw Math::Exception{
          "Math::dotTranspose(const MatrixBase<T>&, const MatrixBase<T>&)",
          "Can't compute the \"transposed\" dot product of two matrices where "
          "the first matrix's row number isn't the same as the second matrix's "
          "row number (the first was rotated)"};

    Matrix<T> result{ma.cols(), mb.cols()};

    for (size_t i{}; i < ma.cols(); ++i) {
      for (size_t j{}; j < mb.cols(); ++j) {
        T sum{};
        for (size_t k{}; k < ma.rows(); ++k)
          sum += ma[k, i] * mb[k, j];
        result[i, j] = sum;
      }
    }

    return result;
  } else {
    if (ma.cols() != mb.cols())
      throw Math::Exception{
          "Math::dotTranspose(const MatrixBase<T>&, const MatrixBase<T>&)",
          "Can't compute the \"transposed\" dot product of two matrices where "
          "the first matrix's col number isn't the same as the second matrix's "
          "col number (the second was rotated)"};

    Matrix<T> result{ma.rows(), mb.rows()};

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
}

template <typename T>
Vector<T> operator+(const VectorBase<T> &a, const VectorBase<T> &b) {
  if (a.size() != b.size())
    throw Math::Exception{
        "Math::operator+(VectorBase<T>, const VectorBase<T>&)",
        "Unable to add two vectors of different sizes"};

  Vector<T> result(a.size());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

  return result;
}

template <typename T>
Matrix<T> operator+(const MatrixBase<T> &m, const VectorBase<T> &v) {
  if (m.cols() == v.size()) { // row wise addition
    Matrix<T> result{m.rows(), m.cols()};
    for (size_t i{}; i < m.rows(); ++i)
      for (size_t j{}; j < m.cols(); ++j)
        result[i, j] = m[i, j] + v[j];

    return result;
  }

  if (m.rows() == v.size()) { // column wise addition
    Matrix<T> result{m.rows(), m.cols()};
    for (size_t i{}; i < m.rows(); ++i)
      for (size_t j{}; j < m.cols(); ++j)
        result[i, j] = m[i, j] + v[i];

    return result;
  }

  throw Math::Exception{
      "Math::operator+(const MatrixBase<T>&, const VectorBase<T>&)",
      "Can't add matrix and vector where their sizes don't match for row or "
      "column wise addition"};
}
}; // namespace Math
