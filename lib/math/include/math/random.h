#pragma once

#include <random>

namespace Math::Random {

// Generates a seeded Mersenne Twister engine.
inline std::mt19937 generate();

// Global Mersenne Twister engine.
extern std::mt19937 mt;

// Generates a random double in [min, max].
inline double get(double min, double max);

// Generates a random integer in [min, max].
inline int getInt(int min, int max);

// Generates a random double from normal distribution (mean, stdDev).
inline double getNormal(double mean = 0.0, double normalDiv = 1.0);
} // namespace Math::Random

// Include template function implementations
#include "random.tpp"
