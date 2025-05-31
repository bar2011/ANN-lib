#pragma once

#include <stddef.h>
#include <vector>

namespace Math {

template <typename T> class VectorView;

// Base vector class - pure virtual interface, can't be instantiated. only to
// inherit for other classes
template <typename T> class VectorBase {
public:
  // Single item access - NO BOUNDS CHECKING
  virtual const T &operator[](size_t index) const = 0;
  // Single item access - WITH BOUNDS CHECKING
  virtual const T &at(size_t index) const = 0;

  // Returns a view of the entire vector
  virtual std::shared_ptr<VectorView<T>> view() const = 0;

  // Returns a view of a range of indicies from the vector
  // Included values are in the range of [start, end)
  // Throws if end > size or start >= end
  virtual std::shared_ptr<VectorView<T>> view(size_t start,
                                              size_t end) const = 0;

  // Getters
  virtual size_t size() const = 0;
  virtual const std::vector<T> &data() const = 0;
};
} // namespace Math
