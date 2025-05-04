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

std::function<double(double, double)> Activation::getBackward() {
  switch (type) {
  case Linear:
    return [](double out, double d) { return d; };
  case Step:
    return [](double out, double d) { return 0; };
  case ReLU:
    return [](double out, double d) { return d * ((out > 0) ? 1 : 0); };
  case Sigmoid:
    return [](double out, double d) { return d * out * (1 - out); };
  }
}

Activation::Activation(Activation &&other) noexcept
    : type{other.type}, args{std::move(other.args)} {}
} // namespace ANN
