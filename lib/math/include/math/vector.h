#pragma once

#include <functional>
#include <vector>

namespace Math {

// Forward declaration of Matrix class
template <typename T> class Matrix;

// Forward declaration of Vector class
template <typename T> class Vector;

// Forward declaration of operator+
template <typename T>
Vector<T> operator+(const Vector<T> &a, const Vector<T> &b);

// Forward-declaratio of operator+
template <typename T>
Matrix<T> operator+(const Matrix<T> &m, const Vector<T> &v);

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

  // Element-wise add operation between two vectors of the same type
  friend Vector<T> operator+ <>(const Vector<T> &a, const Vector<T> &b);
  // Column or row vector addition (depends on the given sizes)
  friend Matrix<T> operator+ <>(const Matrix<T> &m, const Vector<T> &v);

  size_t size() const { return m_size; }

  T &operator[](int index) { return m_data[index]; }
  T operator[](int index) const { return m_data[index]; }

private:
  std::vector<T> m_data{};
  size_t m_size{};
};
}; // namespace Math

// Include template function implementation file
#include "vector.tpp"
