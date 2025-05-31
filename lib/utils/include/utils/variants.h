#pragma once

namespace Utils {
// Helper template for std::visit
// Credit: cppreference.com
template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
} // namespace Utils
