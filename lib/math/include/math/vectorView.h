#pragma once

#include "vectorBase.h"

#include <vector>

namespace Math {

template <typename T> class Vector;

template <typename T> class Matrix;

template <typename T> class MatrixView;

// Class which mimics Math::vector class, but holds a reference to a data vector
// instead of the data itself. Hence, it has no ownership of the data it holds.
template <typename T> class VectorView : public VectorBase<T> {
public:
  VectorView() = default;

  VectorView(const VectorView &other) = default;
  VectorView(VectorView &&other) = default;

  VectorView &operator=(const VectorView &other) = default;
  VectorView &operator=(VectorView &&other) = default;

  // Single item access - NO BOUNDS CHECKING
  const T &operator[](size_t index) const;
  // Single item access - WITH BOUNDS CHECKING
  const T &at(size_t index) const;

  // Get view of the entire vector (so returns *this)
  const VectorView<T> view() const;

  // Returns a view of a range of indices from the vector
  // Included values are in the range of [start, end)
  // Throws if end > size or start >= end
  const VectorView<T> view(size_t start, size_t end) const;

  // Getters
  size_t size() const { return m_size; }
  const std::vector<T> &data() const { return *m_data; }

  friend Vector<T>;

  friend Matrix<T>;

  friend MatrixView<T>;

private:
  VectorView(size_t start, size_t size, const std::vector<T> &data)
      : m_data{&data}, m_start{start}, m_size{size} {}

  const std::vector<T> *m_data{nullptr};
  size_t m_start{};
  size_t m_size{};
};
} // namespace Math

#include "vectorView.tpp"
