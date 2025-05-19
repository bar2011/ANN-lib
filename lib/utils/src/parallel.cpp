#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace Utils {
namespace Parallel {

void parallelFor(size_t loopLength, std::function<void(size_t)> innerLoop,
                 size_t threadCount) {
  if (loopLength == 0)
    return;

  size_t availableThreads{
      (threadCount > 0)
          ? threadCount
          : std::clamp<size_t>(
                static_cast<size_t>(std::thread::hardware_concurrency()), 1,
                loopLength)};
  size_t chunkBaseSize{loopLength / availableThreads};
  size_t numChunkSizeIncrements{loopLength -
                                (chunkBaseSize * availableThreads)};

  std::vector<std::jthread> threads{};

  size_t currentStart{};
  size_t currentEnd{};
  for (size_t thread{}; thread < availableThreads; ++thread) {
    currentEnd = std::min(currentEnd + chunkBaseSize +
                              ((thread < numChunkSizeIncrements) ? 1 : 0),
                          loopLength);

    threads.emplace_back(std::jthread([currentStart, currentEnd, &innerLoop]() {
      for (size_t i{currentStart}; i < currentEnd; ++i)
        innerLoop(i);
    }));

    currentStart = currentEnd;
  }
}
} // namespace Parallel
} // namespace Utils
