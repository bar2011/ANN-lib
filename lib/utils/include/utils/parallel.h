#pragma once

#include <functional>

namespace Utils {
namespace Parallel {

// Runs a loop evenly between available threads (or passed in value if non-zero)
// Function input - index of current iteration
void parallelFor(size_t loopLength, std::function<void(size_t)> innerLoop,
                 size_t threadCount = 0);

} // namespace Parallel
} // namespace Utils
