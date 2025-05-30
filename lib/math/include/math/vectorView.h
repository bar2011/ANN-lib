#pragma once

#include "vectorBase.h"

#include <vector>

namespace Math {

template <typename T> class Vector;

template <typename T> class Matrix;

// Class which mimics Math::vector class, but holds a reference to a data vector
// instead of the data itself. Hence, it has no ownership of the data it holds.
template <typename T> class VectorView : public VectorBase<T> {
public:
  VectorView() = delete;

  // Single item access - NO BOUNDS CHECKING
  const T &operator[](size_t index) const;
  // Single item access - WITH BOUNDS CHECKING
  const T &at(size_t index) const;

  // Returns a view of a range of indicies from the vector
  // Included values are in the range of [start, end)
  // Throws if end > size or start >= end
  std::shared_ptr<VectorView<T>> view(size_t start, size_t end) const;

  // Getters
  size_t size() const { return m_size; }
  const std::vector<T> &data() const { return m_data; }

  friend Vector<T>;

  friend Matrix<T>;

private:
  VectorView(size_t start, size_t size, const std::vector<T> &data)
      : m_data{data}, m_start{start}, m_size{size} {}

  const std::vector<T> &m_data{};
  size_t m_start{};
  size_t m_size{};
};
} // namespace Math

#include "vectorView.tpp"
