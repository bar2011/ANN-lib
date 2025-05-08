#include "ann/activation.h"
#include "ann/exception.h"

#include <cmath>

namespace ANN {
std::function<float(float)> Activation::getForward() {
  switch (type) {
  case Linear:
    return [](float x) { return x; };
  case Step:
    return [](float x) { return (x > 0) ? 1 : 0; };
  case ReLU:
    return [](float x) { return (x > 0) ? x : 0; };
  case Sigmoid:
    return [](float x) { return 1 / (1 + std::exp(-x)); };
  case LeakyReLU:
    auto c{args[0]};
    return [c](float x) { return (x > 0) ? x : (c * x); };
  }
  throw ANN::Exception{"ANN::Activation::getForward()",
                       "Unknown or unsupported activation type entered"};
}

std::function<float(float, float)> Activation::getBackward() {
  switch (type) {
  case Linear:
    return [](float out, float d) { return d; };
  case Step:
    return [](float out, float d) { return 0; };
  case ReLU:
    return [](float out, float d) { return d * ((out > 0) ? 1 : 0); };
  case Sigmoid:
    return [](float out, float d) { return d * out * (1 - out); };
  case LeakyReLU:
    auto c{args[0]};
    return [c](float out, float d) { return d * ((out > 0) ? 1 : c); };
  }
  throw ANN::Exception{"ANN::Activation::getBackward()",
                       "Unknown or unsupported activation type entered"};
}

Activation::Activation(Activation &&other) noexcept
    : type{other.type}, args{std::move(other.args)} {}
} // namespace ANN
