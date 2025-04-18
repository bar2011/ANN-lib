#pragma once

#include <functional>
#include <vector>

namespace Math {

// Simple vector class with access to mathematical operations
// T = the data type the vector holds
template <typename T> class Vector {
public:
  // Default constructor
  Vector() : m_size{0} {};

  // Create 0-filled vector of given length
  Vector(const size_t size) : m_size{size}, m_data(size) {};

  // Create data from given function output
  Vector(const size_t size, std::function<T()> gen);

  // Create data filled vector of given length
  Vector(const size_t size, const T *const data);

  // Copy constructor
  Vector(const Vector &other);

  // Move constructor
  Vector(Vector &&other) noexcept;

  // Copy assignment
  Vector &operator=(const Vector &other);

  // Move assignment
  Vector &operator=(Vector &&other) noexcept;

  size_t size() const { return m_size; }

  T &operator[](size_t index) { return m_data[index]; }
  const T &operator[](size_t index) const { return m_data[index]; }

  T *begin() { return m_data.begin().base(); }
  T *end() { return m_data.end().base(); }

private:
  std::vector<T> m_data{};
  size_t m_size{};
};
}; // namespace Math

// Include template function implementation file
#include "vector.tpp"
