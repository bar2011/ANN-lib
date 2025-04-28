
#pragma once

#include "vectorBase.h"
#include <vector>

namespace Math {

template <typename T> class Vector;

// Class which mimics Math::vector class, but holds a reference to a data vector
// instead of the data itself. Hence, it has no ownership of the data it holds.
template <typename T> class VectorView : public VectorBase<T> {
public:
  VectorView() = delete;

  // Get specific item from vector
  const T &operator[](size_t index) const;

  // Getters
  size_t size() const { return m_size; }

  friend Vector<T>;

private:
  VectorView(size_t start, size_t size, const std::vector<T> &data)
      : m_data{data}, m_start{start}, m_size{size} {}

  const std::vector<T> &m_data{};
  size_t m_start{};
  size_t m_size{};
};
} // namespace Math

#include "vectorView.tpp"
