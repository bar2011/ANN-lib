#pragma once

#include "vectorBase.h"
#include "vectorView.h"

#include <functional>
#include <vector>

namespace Math {
// Simple vector class with access to mathematical operations
// T = the data type the vector holds
template <typename T> class Vector : public VectorBase<T> {
public:
  // Default constructor
  Vector() = default;

  // Create 0-filled vector of given length
  Vector(const size_t size) : m_data(size) {};

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

  // Fill the vector with values from the generator function
  // gen input - a pointer to the item to be filled
  void fill(std::function<void(T *)> gen);

  // Transform the current vector using the given function
  // gen's inputs = a pointer to a value from the current vector, and the
  // corresponding value from v. Both vectors must be the same size
  void transform(const VectorBase<T> &v,
                 std::function<void(T *, const T *)> gen);

  // Get specific item from vector
  T &operator[](size_t index);
  const T &operator[](size_t index) const;

  // Get view of the entire vector
  std::shared_ptr<VectorView<T>> view() const;

  // Returns a view of a range of indicies from the vector
  // Included values are in the range of [start, end)
  // Throws if end > size or start >= end
  std::shared_ptr<VectorView<T>> view(size_t start, size_t end) const;

  // Getters
  size_t size() const { return m_data.size(); }
  std::vector<T> &data() { return m_data; }
  const std::vector<T> &data() const { return m_data; }

  friend Matrix<T>;

private:
  std::vector<T> m_data{};
};
}; // namespace Math

// Include template function implementation file
#include "vector.tpp"
