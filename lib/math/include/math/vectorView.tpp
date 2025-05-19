#pragma once

#include "math/exception.h"
#include "vectorView.h"

namespace Math {
template <typename T> const T &VectorView<T>::operator[](size_t index) const {
  return m_data[m_start + index];
}

template <typename T> const T &VectorView<T>::at(size_t index) const {
  if (index >= m_size || index + m_start >= m_data.size())
    throw Math::Exception{"VectorView<T>::at(size_t) const",
                          "Given index out of bounds"};

  return m_data[m_start + index];
}
} // namespace Math
