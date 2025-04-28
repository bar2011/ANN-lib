#pragma once

#include "math/exception.h"
#include "vectorView.h"

namespace Math {
template <typename T> const T &VectorView<T>::operator[](size_t index) const {
  if (index >= m_size)
    throw Exception{"VectorView<T>::operator[](size_t) const",
                    "Given index out of bounds"};

  return m_data[index];
}
} // namespace Math
