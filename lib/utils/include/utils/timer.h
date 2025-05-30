#pragma once

#include <chrono>

namespace Utils {

// Credit: https://learncpp.com
class Timer {
public:
  // Reset clock
  void reset() { m_beg = Clock::now(); }

  // Acquire elapsed time since last reset / initialization
  double elapsed() const {
    return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
  }

private:
  // Type aliases to make accessing nested type easier
  using Clock = std::chrono::high_resolution_clock;
  using Second = std::chrono::duration<double, std::ratio<1>>;

  std::chrono::time_point<Clock> m_beg{Clock::now()};
};

} // namespace Utils
