#pragma once

#include <chrono>
#include <random>

namespace Math::Random {
// Generates a seeded Mersenne Twister engine.
inline std::mt19937 generate() {
  std::random_device rd{};
  std::seed_seq ss{
      static_cast<std::seed_seq::result_type>(
          std::chrono::steady_clock::now().time_since_epoch().count()),
      rd(),
      rd(),
      rd(),
      rd(),
      rd(),
      rd(),
      rd()};

  return std::mt19937{ss};
}

// Define the global variable.
std::mt19937 mt{generate()};

// Generates a random double in [min, max].
inline double get(double min, double max) {
  return std::uniform_real_distribution{min, max}(mt);
}

// Generates a random integer in [min, max].
inline int getInt(int min, int max) {
  return std::uniform_int_distribution{min, max}(mt);
}

// Generates a random double from normal distribution (mean, stdDev).
inline double getNormal(double mean, double normalDiv) {
  return std::normal_distribution<double>{mean, normalDiv}(mt);
}
} // namespace Math::Random
