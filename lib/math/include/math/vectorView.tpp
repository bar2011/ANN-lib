#pragma once

#include "math/exception.h"
#include "utils/exceptions.h"
#include "vectorView.h"

namespace Math {
template <typename T> const T &VectorView<T>::operator[](size_t index) const {
  return data()[m_start + index];
}

template <typename T> const T &VectorView<T>::at(size_t index) const {
  if (index >= m_size || index + m_start >= data().size())
    throw Math::Exception{CURRENT_FUNCTION, "Given index out of bounds"};

  return data()[m_start + index];
}

template <typename T> const VectorView<T> VectorView<T>::view() const {
  return *this;
}

template <typename T>
const VectorView<T> VectorView<T>::view(size_t start, size_t end) const {
  if (end > data().size() - m_start)
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Unable to create view of vector.\nGiven end is outside of "
        "the vector."};
  if (start >= end)
    throw Math::Exception{
        CURRENT_FUNCTION,
        "Unable to create view of vector.\nGiven start is after "
        "the given end."};

  // Can't use make_shared because it uses a private constructor
  return VectorView<T>{m_start + start, m_start + end - start, *m_data};
}
} // namespace Math
