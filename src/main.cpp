#include "math/matrix.h"
#include "mnist/loader.h"

#include "ann/layers/dense.h"

#include <iostream>
#include <stdexcept>

#include <chrono>

class Timer {
private:
  // Type aliases to make accessing nested type easier
  using Clock = std::chrono::high_resolution_clock;
  using Second = std::chrono::duration<double, std::ratio<1>>;

  std::chrono::time_point<Clock> m_beg{Clock::now()};

public:
  void reset() { m_beg = Clock::now(); }

  double elapsed() const {
    return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
  }
};

template <typename T> void printMatrix(const Math::Matrix<T> &m) {
  for (size_t row{}; row < m.rows(); ++row) {
    for (size_t col{}; col < m.cols(); ++col)
      std::cout << "\033[38;2;" << m[row, col] << ';' << m[row, col] << ';'
                << m[row, col] << "m█\033[0m";
    std::cout << '\n';
  }
}

int main() {
  try {
    MNist::Loader *loader{new MNist::Loader(
        "data/train-labels-idx1-ubyte", "data/train-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte", "data/t10k-images-idx3-ubyte")};
    std::array<MNist::Loader::DataPair, 2> data{loader->loadData()};
    // print first image matrix from data training set
    printMatrix(std::get<1>(data[0])[0]);

    // rows and columns in each image
    constexpr int rows{28};
    constexpr int cols{28};

    auto testLayer{new Layer::Dense<rows * cols, 10>()};
    testLayer->forward(*std::get<1>(data[0])[0].flatten());
    auto output{testLayer->output()};

    for (size_t i{}; i < 10; ++i)
      std::cout << output[i] << ' ';

    std::cout << "\nFinished\n";
  } catch (std::runtime_error &e) {
    std::cout << "An error occured: " << e.what() << '\n';
  }

  return 0;
}
