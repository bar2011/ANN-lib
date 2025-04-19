#include "ann/activation.h"

#include <cmath>

namespace ANN {
std::function<double(double)> Activation::getForward() {
  switch (type) {
  case Linear:
    return [](double x) { return x; };
  case Step:
    return [](double x) { return (x > 0) ? 1 : 0; };
  case ReLU:
    return [](double x) { return (x > 0) ? x : 0; };
  case Sigmoid:
    return [](double x) { return 1 / (1 + std::exp(-x)); };
  }
}

Activation::Activation(Activation &&other) noexcept
    : type{other.type}, args{std::move(other.args)} {}
} // namespace ANN
