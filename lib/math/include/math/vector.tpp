// DO NOT INCLUDE THIS FILE - include vector.h instead
#pragma once

#include "vector.h"

#include "exception.h"
#include <algorithm>

namespace Math {

template <typename T>
Vector<T>::Vector(const size_t size, std::function<T()> gen)
    : m_size{size}, m_data(size) {
  std::generate_n(m_data.begin(), size, gen);
}

template <typename T>
Vector<T>::Vector(const size_t size, const T *data)
    : m_size{size}, m_data(size) {
  std::copy_n(data, size, m_data.begin());
}

template <typename T>
Vector<T>::Vector(const Vector &other)
    : m_size{other.m_size}, m_data(other.m_size) {
  std::copy_n(other.m_data.begin(), m_size, m_data.begin());
}

template <typename T>
Vector<T>::Vector(Vector &&other) noexcept
    : m_size{other.m_size}, m_data{std::move(other.m_data)} {
  other.m_size = 0;
}

template <typename T> Vector<T> &Vector<T>::operator=(const Vector &other) {
  if (&other != this) {
    m_size = other.m_size;
    m_data.resize(other.m_size);
    std::copy_n(other.m_data.begin(), m_size, m_data.begin());
  }
  return *this;
}

template <typename T> Vector<T> &Vector<T>::operator=(Vector &&other) noexcept {
  if (&other != this) {
    m_data = std::move(other.m_data);
    m_size = other.m_size;
    other.m_size = 0;
  }
  return *this;
}

} // namespace Math
