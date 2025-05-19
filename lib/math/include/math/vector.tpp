#pragma once

#include "math/exception.h"
#include "vector.h"

#include "utils/parallel.h"

#include <algorithm>

namespace Math {

template <typename T>
Vector<T>::Vector(const size_t size, std::function<T()> gen) : m_data(size) {
  std::generate_n(m_data.begin(), size, gen);
}

template <typename T>
Vector<T>::Vector(const size_t size, const T *data) : m_data(size) {
  std::copy_n(data, size, m_data.begin());
}

template <typename T>
Vector<T>::Vector(const Vector &other) : m_data(other.m_data.size()) {
  std::copy_n(other.m_data.begin(), m_data.size(), m_data.begin());
}

template <typename T>
Vector<T>::Vector(Vector &&other) noexcept : m_data{std::move(other.m_data)} {
  other.m_data.resize(0);
}

template <typename T> Vector<T> &Vector<T>::operator=(const Vector &other) {
  if (&other != this) {
    m_data.size() = other.m_data.size();
    m_data.resize(other.m_data.size());
    std::copy_n(other.m_data.begin(), m_data.size(), m_data.begin());
  }
  return *this;
}

template <typename T> Vector<T> &Vector<T>::operator=(Vector &&other) noexcept {
  if (&other != this) {
    m_data = std::move(other.m_data);
    m_data.size() = other.m_data.size();
    other.m_data.size() = 0;
  }
  return *this;
}

template <typename T>
void Vector<T>::fill(std::function<void(T *)> gen,
                     std::optional<bool> parallelize, size_t cost) {
  Utils::Parallel::dynamicParallelFor(
      cost, m_data.size(), [gen, &data = m_data](size_t i) { gen(&data[i]); },
      parallelize);
}

template <typename T>
void Vector<T>::transform(const VectorBase<T> &v,
                          std::function<void(T *, const T *)> gen) {
  for (size_t i{}; i < m_data.size(); ++i)
    gen(&m_data[i], &v[i]);
}

template <typename T>
void Vector<T>::transform(const VectorBase<T> &va, const VectorBase<T> &vb,
                          std::function<void(T *, const T *, const T *)> gen) {
  for (size_t i{}; i < m_data.size(); ++i)
    gen(&m_data[i], &va[i], &vb[i]);
}

template <typename T> T &Vector<T>::operator[](size_t index) {
  if (index >= m_data.size())
    throw Math::Exception{"Vector<T>::operator[](size_t)",
                          "Given index out of bounds"};

  return m_data[index];
}

template <typename T> const T &Vector<T>::operator[](size_t index) const {
  if (index >= m_data.size())
    throw Math::Exception{"Vector<T>::operator[](size_t) const",
                          "Given index out of bounds"};

  return m_data[index];
}

template <typename T> std::shared_ptr<VectorView<T>> Vector<T>::view() const {
  return std::make_shared<VectorView<T>>(0, m_data.size(), m_data);
}

template <typename T>
std::shared_ptr<VectorView<T>> Vector<T>::view(size_t start, size_t end) const {
  if (end > m_data.size())
    throw Math::Exception{
        "Vector<T>::view(size_t, size_t) const",
        "Unable to create view of vector.\nGiven end is outside of "
        "the vector."};
  if (start >= end)
    throw Math::Exception{
        "Vector<T>::view(size_t, size_t) const",
        "Unable to create view of vector.\nGiven start is after "
        "the given end."};

  // Can't use make_shared because it uses a private constructor
  return std::shared_ptr<VectorView<T>>{
      new VectorView<T>{start, end - start, m_data}};
}
} // namespace Math
