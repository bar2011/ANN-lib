#pragma once

#include <functional>

namespace Utils {
namespace Parallel {

// Runs a loop evenly between available threads (or passed in value if non-zero)
// loopLength - length of loop needed to be parallelized
// innerLoop - code to be ran in every loop iteration. Passed in value is the
//             iteration. function shouldn't rely on any past iterations, or the
//             order the iterations are ran.
// threadCount (optional) - customize the number of threads which'll be
//                          initialized
void parallelFor(size_t loopLength, std::function<void(size_t)> innerLoop,
                 size_t threadCount = 0);

// If cost passes a threshold, run parallelFor() with the provided inputs.
// Otherwise, run the innerLoop in a simple for loop fashion.
// cost - estimated cost of the iterations in innerLoop, where 1 represents a
//        single addition
// loopLength - length of loop needed to be parallelized
// innerLoop - code to be ran in every loop iteration. Passed in value is the
//             iteration. function shouldn't rely on any past iterations, or the
//             order the iterations are ran.
// parallelize (optional) - if not empty, overrides the cost calculation in the
//                       choice of parallelizing or not.
// threadCount (optional) - customize the number of threads which'll be
//                          initialized.
void dynamicParallelFor(size_t cost, size_t loopLength,
                        std::function<void(size_t)> innerLoop,
                        std::optional<bool> parallelize = std::nullopt,
                        size_t threadCount = 0);

} // namespace Parallel
} // namespace Utils
